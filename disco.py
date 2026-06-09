import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCFLayer(nn.Module):
    def __init__(self, embedding_dim, dropout, negative_slope):
        super().__init__()
        self.linear_sum = nn.Linear(embedding_dim, embedding_dim)
        self.linear_bi = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope

    def forward(self, self_embedding, neighbor_embeddings, edge_weights):
        messages = self.linear_sum(neighbor_embeddings)
        messages = messages + self.linear_bi(
            neighbor_embeddings * self_embedding.unsqueeze(1)
        )
        messages = (messages * edge_weights.unsqueeze(-1)).sum(dim=1)
        output = self.linear_sum(self_embedding) + messages
        output = F.leaky_relu(output, negative_slope=self.negative_slope)
        return self.dropout(output)


class DisentangledGraphEncoder(nn.Module):
    """One shared NGCF layer followed by K intent-specific NGCF channels."""

    def __init__(
        self,
        num_users,
        graph,
        embedding_dim,
        num_intents,
        dropout=0.1,
        negative_slope=0.05,
    ):
        super().__init__()
        self.num_intents = num_intents
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(graph["num_items"], embedding_dim)
        self.shared_layer = NGCFLayer(
            embedding_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.intent_layers = nn.ModuleList(
            [
                NGCFLayer(
                    embedding_dim,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
                for _ in range(num_intents)
            ]
        )

        self.register_buffer(
            "user_neighbors", graph["user_neighbors"], persistent=False
        )
        self.register_buffer(
            "item_neighbors", graph["item_neighbors"], persistent=False
        )
        self.register_buffer("user_degree", graph["user_degree"], persistent=False)
        self.register_buffer("item_degree", graph["item_degree"], persistent=False)
        self.item_offset = graph["item_offset"]
        self.num_items = graph["num_items"]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @staticmethod
    def _safe_neighbors(neighbors):
        mask = neighbors >= 0
        return neighbors.clamp_min(0), mask

    def _user_edge_weights(self, user_ids, item_ids, mask):
        user_degree = self.user_degree[user_ids].unsqueeze(1)
        item_degree = self.item_degree[item_ids]
        weights = torch.rsqrt(user_degree * item_degree)
        return weights * mask

    def _item_edge_weights(self, item_ids, user_ids, mask):
        item_degree = self.item_degree[item_ids].unsqueeze(1)
        user_degree = self.user_degree[user_ids]
        weights = torch.rsqrt(item_degree * user_degree)
        return weights * mask

    def shared_users(self, user_ids):
        neighbors, mask = self._safe_neighbors(self.user_neighbors[user_ids])
        self_embedding = self.user_embedding(user_ids)
        neighbor_embeddings = self.item_embedding(neighbors)
        weights = self._user_edge_weights(user_ids, neighbors, mask)
        return self.shared_layer(self_embedding, neighbor_embeddings, weights)

    def shared_items(self, item_ids):
        neighbors, mask = self._safe_neighbors(self.item_neighbors[item_ids])
        self_embedding = self.item_embedding(item_ids)
        neighbor_embeddings = self.user_embedding(neighbors)
        weights = self._item_edge_weights(item_ids, neighbors, mask)
        return self.shared_layer(self_embedding, neighbor_embeddings, weights)

    def encode_users(self, user_ids):
        neighbors, mask = self._safe_neighbors(self.user_neighbors[user_ids])
        shared_self = self.shared_users(user_ids)
        shared_neighbors = self.shared_items(neighbors.reshape(-1))
        shared_neighbors = shared_neighbors.view(
            neighbors.size(0), neighbors.size(1), -1
        )
        weights = self._user_edge_weights(user_ids, neighbors, mask)
        intents = [
            layer(shared_self, shared_neighbors, weights)
            for layer in self.intent_layers
        ]
        return torch.stack(intents, dim=1)

    def encode_items(self, global_item_ids):
        item_ids = global_item_ids - self.item_offset
        if torch.any(item_ids < 0) or torch.any(item_ids >= self.num_items):
            raise ValueError("item id is outside this encoder's domain")

        neighbors, mask = self._safe_neighbors(self.item_neighbors[item_ids])
        shared_self = self.shared_items(item_ids)
        shared_neighbors = self.shared_users(neighbors.reshape(-1))
        shared_neighbors = shared_neighbors.view(
            neighbors.size(0), neighbors.size(1), -1
        )
        weights = self._item_edge_weights(item_ids, neighbors, mask)
        intents = [
            layer(shared_self, shared_neighbors, weights)
            for layer in self.intent_layers
        ]
        return torch.stack(intents, dim=1)


class DisCo(nn.Module):
    def __init__(
        self,
        num_users,
        source_graph,
        target_graph,
        embedding_dim=128,
        num_intents=4,
        alpha=0.1,
        beta=0.3,
        gamma=0.01,
        contrast_weight=0.3,
        temperature=0.2,
        random_walk_steps=3,
        ema_decay=0.99,
        dropout=0.1,
        negative_slope=0.05,
    ):
        super().__init__()
        if num_intents < 1:
            raise ValueError("num_intents must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if random_walk_steps < 1:
            raise ValueError("random_walk_steps must be positive")
        self.num_intents = num_intents
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.contrast_weight = contrast_weight
        self.temperature = temperature
        self.random_walk_steps = random_walk_steps
        self.ema_decay = ema_decay

        encoder_args = dict(
            num_users=num_users,
            embedding_dim=embedding_dim,
            num_intents=num_intents,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.source_encoder = DisentangledGraphEncoder(
            graph=source_graph, **encoder_args
        )
        self.target_encoder = DisentangledGraphEncoder(
            graph=target_graph, **encoder_args
        )
        self.source_momentum_encoder = copy.deepcopy(self.source_encoder)
        self.target_momentum_encoder = copy.deepcopy(self.target_encoder)
        self._freeze_momentum_encoders()

        self.cross_domain_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.source_prototypes = nn.Parameter(
            torch.empty(num_intents, embedding_dim)
        )
        self.target_prototypes = nn.Parameter(
            torch.empty(num_intents, embedding_dim)
        )
        nn.init.xavier_uniform_(self.source_prototypes)
        nn.init.xavier_uniform_(self.target_prototypes)

    def _freeze_momentum_encoders(self):
        for encoder in (
            self.source_momentum_encoder,
            self.target_momentum_encoder,
        ):
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.source_momentum_encoder.eval()
        self.target_momentum_encoder.eval()
        return self

    @torch.no_grad()
    def update_momentum_encoders(self):
        pairs = (
            (self.source_encoder, self.source_momentum_encoder),
            (self.target_encoder, self.target_momentum_encoder),
        )
        for online, target in pairs:
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.data.mul_(self.ema_decay).add_(
                    online_parameter.data, alpha=1.0 - self.ema_decay
                )

    def _intent_prior(self, user_intents, prototypes):
        user_intents = F.normalize(user_intents, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        logits = (user_intents * prototypes.unsqueeze(0)).sum(dim=-1)
        return F.softmax(logits, dim=1)

    def recommendation_loss(self, domain, users, positive_items, negative_items):
        if domain == "src":
            encoder = self.source_encoder
            prototypes = self.source_prototypes
        elif domain == "tgt":
            encoder = self.target_encoder
            prototypes = self.target_prototypes
        else:
            raise ValueError("domain must be 'src' or 'tgt'")

        items = torch.cat([positive_items, negative_items])
        user_intents = encoder.encode_users(users)
        item_intents = encoder.encode_items(items)
        positive_intents, negative_intents = item_intents.chunk(2, dim=0)
        prior = self._intent_prior(user_intents, prototypes)
        positive_logits = (
            prior * (user_intents * positive_intents).sum(dim=-1)
        ).sum(dim=1)
        negative_logits = (
            prior * (user_intents * negative_intents).sum(dim=-1)
        ).sum(dim=1)
        return F.softplus(-positive_logits).mean() + F.softplus(
            negative_logits
        ).mean()

    def _random_walk_target(self, target_intents):
        batch_size = target_intents.size(0)
        identity = torch.eye(
            batch_size,
            dtype=target_intents.dtype,
            device=target_intents.device,
        )
        targets = []
        for intent in range(self.num_intents):
            embedding = target_intents[:, intent]
            distance = torch.cdist(embedding, embedding, p=2)
            affinity = torch.exp(-distance / self.temperature)
            transition = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(
                1e-12
            )
            walk = torch.linalg.matrix_power(
                transition, self.random_walk_steps
            )
            targets.append(self.alpha * identity + (1.0 - self.alpha) * walk)
        return torch.stack(targets, dim=1)

    def _intra_domain_loss(self, online_intents, target_intents):
        with torch.no_grad():
            target_distribution = self._random_walk_target(target_intents)
        online = F.normalize(online_intents, dim=-1)
        target = F.normalize(target_intents, dim=-1)
        logits = torch.einsum("ikd,jkd->ikj", online, target)
        log_probabilities = F.log_softmax(
            logits / self.temperature, dim=-1
        )
        return -(target_distribution * log_probabilities).sum(dim=-1).mean()

    @staticmethod
    def _orthogonality_loss(intents):
        if intents.size(0) < 2:
            return intents.new_zeros(())
        centered = intents - intents.mean(dim=0, keepdim=True)
        centered = centered / (
            centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        )
        covariance = torch.einsum("bkd,bke->kde", centered, centered)
        covariance = covariance / intents.size(0)
        identity = torch.eye(
            intents.size(-1), device=intents.device, dtype=intents.dtype
        )
        return (covariance - identity.unsqueeze(0)).square().mean()

    def contrastive_loss(self, users):
        source_online = self.source_encoder.encode_users(users)
        target_online = self.target_encoder.encode_users(users)
        with torch.no_grad():
            source_target = self.source_momentum_encoder.encode_users(users)
            target_target = self.target_momentum_encoder.encode_users(users)

        intra = self._intra_domain_loss(
            source_online, source_target
        ) + self._intra_domain_loss(target_online, target_target)
        orthogonal = self._orthogonality_loss(
            source_online
        ) + self._orthogonality_loss(target_online)
        inter = self._inter_domain_loss(source_online, target_target)
        return self.beta * inter + (1.0 - self.beta) * (
            intra + self.gamma * orthogonal
        )

    def _inter_domain_loss(self, source_intents, target_intents):
        mapped_source = self.cross_domain_decoder(source_intents)
        mapped_source = F.normalize(mapped_source, dim=-1)
        target_intents = F.normalize(target_intents, dim=-1)

        logits = torch.einsum(
            "ikd,jkd->ikj", mapped_source, target_intents
        )
        log_likelihood = F.log_softmax(
            logits / self.temperature, dim=-1
        )
        prior = self._intent_prior(mapped_source, self.target_prototypes)
        log_prior = torch.log(prior.clamp_min(1e-12))

        with torch.no_grad():
            posterior = F.softmax(
                log_prior.unsqueeze(-1) + log_likelihood, dim=1
            )
            target_similarity = self._random_walk_target(
                target_intents
            ).mean(dim=1)

        expected_log_likelihood = (
            posterior * log_likelihood
        ).sum(dim=1)
        kl_divergence = (
            posterior
            * (
                torch.log(posterior.clamp_min(1e-12))
                - log_prior.unsqueeze(-1)
            )
        ).sum(dim=1)
        elbo = expected_log_likelihood - kl_divergence
        return -(target_similarity * elbo).sum(dim=1).mean()

    def total_loss(self, source_batch, target_batch, overlap_users):
        source_rec = self.recommendation_loss("src", *source_batch)
        target_rec = self.recommendation_loss("tgt", *target_batch)
        contrast = self.contrastive_loss(overlap_users)
        recommendation = source_rec + target_rec
        total = (1.0 - self.contrast_weight) * recommendation
        total = total + self.contrast_weight * contrast
        return total, {
            "source_rec": source_rec.detach(),
            "target_rec": target_rec.detach(),
            "contrast": contrast.detach(),
        }

    def cross_domain_logits(self, users, target_items):
        source_intents = self.source_encoder.encode_users(users)
        mapped_source = self.cross_domain_decoder(source_intents)
        prior = self._intent_prior(mapped_source, self.target_prototypes)

        if target_items.ndim == 1:
            target_intents = self.target_encoder.encode_items(target_items)
            scores = (mapped_source * target_intents).sum(dim=-1)
            return (prior * scores).sum(dim=-1)

        flat_items = target_items.reshape(-1)
        target_intents = self.target_encoder.encode_items(flat_items)
        target_intents = target_intents.view(
            target_items.size(0),
            target_items.size(1),
            self.num_intents,
            self.embedding_dim,
        )
        scores = torch.einsum("bkd,bckd->bck", mapped_source, target_intents)
        return (prior.unsqueeze(1) * scores).sum(dim=-1)

    def cross_domain_pair_loss(self, users, positive_items, negative_items):
        items = torch.stack([positive_items, negative_items], dim=1)
        logits = self.cross_domain_logits(users, items)
        return F.softplus(-logits[:, 0]).mean() + F.softplus(
            logits[:, 1]
        ).mean()
