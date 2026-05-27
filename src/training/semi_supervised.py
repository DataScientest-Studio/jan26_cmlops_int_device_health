"""
Semi-supervised learning module for device health monitoring.

Implements K-means clustering with label propagation for scenarios where ground truth
labels are scarce (5-10% of data). Includes automatic K optimization, handling of
unlabeled clusters, and sliding window data selection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SemiSupervisedTrainer:
    """
    Semi-supervised learning trainer using K-means clustering with label propagation.

    This approach is designed for scenarios where:
    - Ground truth labels are scarce (5-10% of data)
    - Multiple failure modes exist (requiring K > 2 clusters)
    - Concept drift requires sliding window approach
    - Some clusters may have zero labeled members

    The training flow:
    1. Optimize K using silhouette score or elbow method
    2. Perform K-means clustering
    3. Assign labels to clusters via majority voting
    4. Handle unlabeled clusters using distance/proximity/heuristics
    5. Train LogisticRegression on propagated labels
    """

    def __init__(
        self,
        k_range: tuple[int, int] = (2, 10),
        k_method: str = "silhouette",
        distance_threshold: float = 2.0,
        knn_neighbors: int = 5,
        use_domain_heuristics: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize semi-supervised trainer.

        Args:
            k_range: Tuple of (min_k, max_k) for K optimization
            k_method: Method for K selection - "silhouette", "elbow", or "calinski"
            distance_threshold: Std deviations from healthy centroid for distance heuristic
            knn_neighbors: Number of neighbors for KNN-based labeling
            use_domain_heuristics: Whether to apply domain-specific rules
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.k_method = k_method
        self.distance_threshold = distance_threshold
        self.knn_neighbors = knn_neighbors
        self.use_domain_heuristics = use_domain_heuristics
        self.random_state = random_state

        self.optimal_k_: int | None = None
        self.kmeans_: KMeans | None = None
        self.cluster_labels_: dict[int, str] | None = None
        self.scaler_ = StandardScaler()

    def optimize_k(
        self, X: np.ndarray, labeled_indices: np.ndarray
    ) -> tuple[int, dict[int, float]]:
        """
        Find optimal number of clusters using silhouette score, elbow method, or CH index.

        Args:
            X: Feature matrix (n_samples, n_features)
            labeled_indices: Boolean mask or indices of labeled samples

        Returns:
            Tuple of (optimal_k, scores_dict)

        Example:
            >>> X = np.random.randn(1000, 10)
            >>> labeled = np.zeros(1000, dtype=bool)
            >>> labeled[:100] = True  # 10% labeled
            >>> optimal_k, scores = trainer.optimize_k(X, labeled)
            >>> print(f"Optimal K: {optimal_k}")
        """
        min_k, max_k = self.k_range

        # Adjust max_k based on number of labeled samples
        n_labeled = np.sum(labeled_indices)
        max_k = min(max_k, n_labeled // 2)  # At least 2 labeled samples per cluster
        max_k = max(max_k, min_k)  # Ensure max_k >= min_k

        if max_k < min_k:
            logger.warning(
                f"Not enough labeled samples ({n_labeled}) for K optimization. Using K={min_k}"
            )
            return min_k, {min_k: 1.0}

        scores = {}
        inertias = {}

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Compute score based on method
            if self.k_method == "silhouette":
                score = silhouette_score(X, cluster_labels)
            elif self.k_method == "calinski":
                score = calinski_harabasz_score(X, cluster_labels)
            elif self.k_method == "elbow":
                score = -kmeans.inertia_  # Negative so higher is better
                inertias[k] = kmeans.inertia_
            else:
                raise ValueError(f"Unknown k_method: {self.k_method}")

            scores[k] = score
            logger.debug(f"K={k}: {self.k_method} score = {score:.4f}")

        # Select K with highest score
        optimal_k = max(scores, key=scores.get)

        # For elbow method, try to detect elbow point
        if self.k_method == "elbow" and len(inertias) > 2:
            optimal_k = self._detect_elbow(inertias)

        logger.info(
            f"Optimal K selected: {optimal_k} using {self.k_method} method "
            f"(score={scores[optimal_k]:.4f})"
        )

        self.optimal_k_ = optimal_k
        return optimal_k, scores

    def _detect_elbow(self, inertias: dict[int, float]) -> int:
        """
        Detect elbow point in inertia curve.

        Uses the "knee/elbow" detection method: finds the point of maximum curvature
        in the inertia vs K curve.

        Args:
            inertias: Dictionary mapping K -> inertia

        Returns:
            K value at elbow point
        """
        k_values = sorted(inertias.keys())
        inertia_values = [inertias[k] for k in k_values]

        # Normalize to [0, 1]
        k_norm = np.linspace(0, 1, len(k_values))
        inertia_norm = (inertia_values - np.min(inertia_values)) / (
            np.max(inertia_values) - np.min(inertia_values) + 1e-10
        )

        # Find point with maximum distance to line from first to last point
        distances = []
        for i in range(len(k_norm)):
            # Distance from point to line
            x0, y0 = k_norm[i], inertia_norm[i]
            x1, y1 = k_norm[0], inertia_norm[0]
            x2, y2 = k_norm[-1], inertia_norm[-1]

            # Point-to-line distance formula
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distance = numerator / (denominator + 1e-10)
            distances.append(distance)

        elbow_index = np.argmax(distances)
        elbow_k = k_values[elbow_index]

        logger.debug(f"Elbow method detected K={elbow_k} (distances={distances})")
        return elbow_k

    def cluster_and_label(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_features_df: pd.DataFrame | None = None,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
        """
        Perform K-means clustering and propagate labels to all samples.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector with -1 for unlabeled, 0/1 for labeled
            X_features_df: Optional DataFrame with feature names (for domain heuristics)
            k: Number of clusters (if None, will optimize)

        Returns:
            Tuple of:
                - cluster_assignments: Cluster ID for each sample
                - propagated_labels: Label (0/1) for each sample
                - cluster_info: Dict with metadata per cluster

        Example:
            >>> X = np.random.randn(1000, 10)
            >>> y = np.full(1000, -1)  # All unlabeled
            >>> y[:100] = np.random.randint(0, 2, 100)  # 10% labeled
            >>> clusters, labels, info = trainer.cluster_and_label(X, y)
            >>> print(f"Propagated {np.sum(y == -1)} unlabeled samples")
        """
        # Identify labeled samples
        labeled_mask = y != -1
        n_labeled = np.sum(labeled_mask)
        n_total = len(y)

        logger.info(
            f"Starting semi-supervised learning: {n_labeled}/{n_total} samples labeled "
            f"({100 * n_labeled / n_total:.1f}%)"
        )

        if n_labeled < 2:
            raise ValueError(
                f"Need at least 2 labeled samples for semi-supervised learning, got {n_labeled}"
            )

        # Optimize K if not provided
        if k is None:
            k, _ = self.optimize_k(X, labeled_mask)
        else:
            self.optimal_k_ = k

        # Perform K-means clustering
        self.kmeans_ = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_assignments = self.kmeans_.fit_predict(X)

        # Assign labels to each cluster via majority voting
        cluster_info = {}
        self.cluster_labels_ = {}

        for cluster_id in range(k):
            cluster_mask = cluster_assignments == cluster_id
            cluster_size = np.sum(cluster_mask)

            # Get labeled samples in this cluster
            cluster_labeled_mask = cluster_mask & labeled_mask
            cluster_labeled_count = np.sum(cluster_labeled_mask)

            info = {
                "size": int(cluster_size),
                "labeled_count": int(cluster_labeled_count),
                "centroid": self.kmeans_.cluster_centers_[cluster_id].tolist(),
            }

            if cluster_labeled_count > 0:
                # Majority vote from labeled samples
                cluster_labels = y[cluster_labeled_mask]
                label_counts = np.bincount(cluster_labels.astype(int))
                majority_label = int(np.argmax(label_counts))
                confidence = label_counts[majority_label] / cluster_labeled_count

                self.cluster_labels_[cluster_id] = majority_label
                info["label"] = majority_label
                info["confidence"] = float(confidence)
                info["method"] = "majority_vote"
                info["is_pseudo_label"] = False

                logger.debug(
                    f"Cluster {cluster_id}: {cluster_size} samples, "
                    f"{cluster_labeled_count} labeled → label={majority_label} "
                    f"(confidence={confidence:.2f})"
                )
            else:
                # Unlabeled cluster - apply heuristics
                label, method = self._handle_unlabeled_cluster(
                    cluster_id=cluster_id,
                    cluster_mask=cluster_mask,
                    X=X,
                    y=y,
                    labeled_mask=labeled_mask,
                    X_features_df=X_features_df,
                )

                self.cluster_labels_[cluster_id] = label
                info["label"] = label
                info["confidence"] = 0.0  # Low confidence for pseudo-labels
                info["method"] = method
                info["is_pseudo_label"] = True

                logger.warning(
                    f"Cluster {cluster_id}: {cluster_size} samples, 0 labeled → "
                    f"assigned label={label} via {method} (PSEUDO-LABEL)"
                )

            cluster_info[cluster_id] = info

        # Propagate cluster labels to all samples
        propagated_labels = np.array([self.cluster_labels_[c] for c in cluster_assignments])

        # CRITICAL: Ensure label diversity (both classes present)
        # This handles edge case where all clusters get the same label
        unique_labels = set(self.cluster_labels_.values())
        if len(unique_labels) < 2:
            logger.warning(
                f"All {k} clusters assigned same label ({list(unique_labels)[0]}). "
                f"Enforcing label diversity to prevent training failure..."
            )
            self._ensure_label_diversity(X, cluster_assignments, cluster_info)
            # Re-propagate with enforced diversity
            propagated_labels = np.array([self.cluster_labels_[c] for c in cluster_assignments])

        n_pseudo = sum(1 for info in cluster_info.values() if info["is_pseudo_label"])
        logger.info(f"Clustering complete: K={k}, {n_pseudo}/{k} clusters pseudo-labeled")

        return cluster_assignments, propagated_labels, cluster_info

    def _handle_unlabeled_cluster(
        self,
        cluster_id: int,
        cluster_mask: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        labeled_mask: np.ndarray,
        X_features_df: pd.DataFrame | None = None,
    ) -> tuple[int, str]:
        """
        Assign label to cluster with no labeled members.

        Tries strategies in order:
        1. Distance to health heuristic
        2. Proximity-based (KNN voting)
        3. Domain-specific heuristics (if available)
        4. Default to unhealthy (conservative)

        Args:
            cluster_id: Cluster index
            cluster_mask: Boolean mask for this cluster
            X: Full feature matrix
            y: Full label vector
            labeled_mask: Boolean mask for labeled samples
            X_features_df: Optional DataFrame with feature names

        Returns:
            Tuple of (label, method_name)
        """
        centroid = self.kmeans_.cluster_centers_[cluster_id]

        # Strategy 1: Distance to health heuristic
        label, success = self._distance_to_health_heuristic(
            centroid, cluster_id, labeled_mask, y, X
        )
        if success:
            return label, "distance_to_health"

        # Strategy 2: Proximity-based (KNN voting)
        label, success = self._proximity_based_labeling(centroid, X, y, labeled_mask)
        if success:
            return label, "knn_voting"

        # Strategy 3: Domain heuristics (if enabled and features available)
        if self.use_domain_heuristics and X_features_df is not None:
            cluster_features = X_features_df[cluster_mask]
            label, success = self._domain_heuristics(cluster_features)
            if success:
                return label, "domain_heuristic"

        # Strategy 4: Conservative default (label as unhealthy)
        logger.warning(
            f"Cluster {cluster_id}: All heuristics failed, defaulting to unhealthy (label=1)"
        )
        return 1, "default_unhealthy"

    def _distance_to_health_heuristic(
        self,
        centroid: np.ndarray,
        cluster_id: int,
        labeled_mask: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
    ) -> tuple[int, bool]:
        """
        Label cluster based on distance to known healthy cluster centroid.

        Logic: If distance > threshold * std_dev, likely unhealthy.

        Args:
            centroid: Centroid of unlabeled cluster
            cluster_id: Cluster ID
            labeled_mask: Boolean mask for labeled samples
            y: Label vector
            X: Feature matrix

        Returns:
            Tuple of (label, success_flag)
        """
        # Find healthy cluster centroid (cluster with most healthy labels)
        healthy_mask = labeled_mask & (y == 0)

        if np.sum(healthy_mask) < 2:
            return 0, False  # Not enough healthy samples

        # Find which cluster has most healthy samples
        healthy_samples = X[healthy_mask]
        healthy_cluster_assignments = self.kmeans_.predict(healthy_samples)
        healthy_cluster_id = np.bincount(healthy_cluster_assignments).argmax()
        healthy_centroid = self.kmeans_.cluster_centers_[healthy_cluster_id]

        # Compute distance
        distance = np.linalg.norm(centroid - healthy_centroid)

        # Compute typical distance scale (std dev of healthy samples from their centroid)
        healthy_distances = np.linalg.norm(healthy_samples - healthy_centroid, axis=1)
        distance_std = np.std(healthy_distances)

        # If distance > threshold * std_dev, consider unhealthy
        threshold_distance = self.distance_threshold * distance_std
        label = 1 if distance > threshold_distance else 0

        logger.debug(
            f"Distance heuristic: cluster {cluster_id} distance={distance:.2f}, "
            f"threshold={threshold_distance:.2f} → label={label}"
        )

        return label, True

    def _proximity_based_labeling(
        self,
        centroid: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        labeled_mask: np.ndarray,
    ) -> tuple[int, bool]:
        """
        Label cluster based on KNN voting from nearest labeled samples.

        Args:
            centroid: Centroid of unlabeled cluster
            X: Feature matrix
            y: Label vector
            labeled_mask: Boolean mask for labeled samples

        Returns:
            Tuple of (label, success_flag)
        """
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]

        if len(X_labeled) < self.knn_neighbors:
            return 0, False  # Not enough labeled samples

        # Fit KNN classifier
        knn = KNeighborsClassifier(n_neighbors=self.knn_neighbors)
        knn.fit(X_labeled, y_labeled)

        # Predict label for centroid
        label = int(knn.predict([centroid])[0])

        logger.debug(f"KNN proximity labeling: K={self.knn_neighbors} → label={label}")

        return label, True

    def _domain_heuristics(self, cluster_features: pd.DataFrame) -> tuple[int, bool]:
        """
        Apply domain-specific rules to label cluster.

        For device health, heuristics might include:
        - High noise_level → unhealthy
        - Low SNR → unhealthy
        - High vibration + high temperature → unhealthy

        Args:
            cluster_features: DataFrame with feature values for cluster members

        Returns:
            Tuple of (label, success_flag)
        """
        # Example heuristics (customize based on your domain knowledge)
        try:
            avg_features = cluster_features.mean()

            # Rule 1: High noise level
            if "noise_level" in avg_features.index:
                if avg_features["noise_level"] > 0.7:  # Threshold
                    logger.debug("Domain heuristic: High noise_level → unhealthy")
                    return 1, True

            # Rule 2: Low SNR
            if "SNR" in avg_features.index and avg_features["SNR"] < 0.3:  # Threshold
                logger.debug("Domain heuristic: Low SNR → unhealthy")
                return 1, True

            # Rule 3: High temperature + vibration
            if (
                "temperature_proxy" in avg_features.index
                and "vibration_proxy" in avg_features.index
            ) and (
                avg_features["temperature_proxy"] > 0.8 and avg_features["vibration_proxy"] > 0.6
            ):
                logger.debug("Domain heuristic: High temp + vibration → unhealthy")
                return 1, True

            # No heuristic triggered → default
            return 0, False

        except Exception as e:
            logger.warning(f"Domain heuristics failed: {e}")
            return 0, False

    def _ensure_label_diversity(
        self,
        X: np.ndarray,
        cluster_assignments: np.ndarray,
        cluster_info: dict[int, dict],
    ) -> None:
        """
        Ensure at least one cluster is assigned each class (0 and 1).

        This prevents LogisticRegression training failure when all propagated
        labels belong to a single class. Uses feature-based heuristics to
        identify which cluster should be flipped.

        Args:
            X: Feature matrix
            cluster_assignments: Cluster ID for each sample
            cluster_info: Dict with metadata per cluster (modified in-place)

        Side Effects:
            Modifies self.cluster_labels_ and cluster_info in-place
        """
        unique_labels = set(self.cluster_labels_.values())

        if len(unique_labels) == 2:
            return  # Already diverse

        # All clusters have the same label - need to flip one
        majority_label = list(unique_labels)[0]
        minority_label = 1 - majority_label

        logger.warning(
            f"All clusters labeled as {majority_label}. "
            f"Searching for cluster to label as {minority_label}..."
        )

        if majority_label == 0:  # All healthy, need to mark one unhealthy
            # Find cluster with highest noise or lowest SNR
            cluster_to_flip = self._find_most_unhealthy_cluster(X, cluster_assignments)
            new_label = 1
            reason = "most_unhealthy_features"

        else:  # All unhealthy, need to mark one healthy
            # Find cluster with lowest noise or highest SNR
            cluster_to_flip = self._find_most_healthy_cluster(X, cluster_assignments)
            new_label = 0
            reason = "most_healthy_features"

        # Flip the label
        old_label = self.cluster_labels_[cluster_to_flip]
        self.cluster_labels_[cluster_to_flip] = new_label

        # Update cluster info
        cluster_info[cluster_to_flip]["label"] = new_label
        cluster_info[cluster_to_flip]["method"] = f"diversity_enforcement_{reason}"
        cluster_info[cluster_to_flip]["is_pseudo_label"] = True
        cluster_info[cluster_to_flip]["diversity_enforced"] = True

        logger.warning(
            f"✓ Diversity enforced: Cluster {cluster_to_flip} "
            f"changed from {old_label} → {new_label} (reason: {reason})"
        )

    def _find_most_unhealthy_cluster(self, X: np.ndarray, cluster_assignments: np.ndarray) -> int:
        """
        Find cluster with most "unhealthy" characteristics.

        Uses heuristics:
        - Highest mean noise_level (feature index 3)
        - Lowest mean SNR (feature index 4)

        Args:
            X: Feature matrix (n_samples, n_features)
                Expected features: [fwhm, peak_height, peak_area, noise_level, snr, peak_center]
            cluster_assignments: Cluster ID for each sample

        Returns:
            Cluster ID with highest unhealthy score
        """
        n_clusters = len(self.cluster_labels_)
        unhealthy_scores = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_samples = X[cluster_mask]

            if len(cluster_samples) == 0:
                continue

            # Feature indices (based on standard feature extraction)
            NOISE_LEVEL_IDX = 3
            SNR_IDX = 4

            # Score based on noise and SNR
            mean_noise = np.mean(cluster_samples[:, NOISE_LEVEL_IDX])
            mean_snr = np.mean(cluster_samples[:, SNR_IDX])

            # Higher noise + lower SNR = more unhealthy
            unhealthy_scores[cluster_id] = mean_noise - 0.1 * mean_snr

        most_unhealthy = int(np.argmax(unhealthy_scores))

        logger.debug(f"Unhealthy scores: {unhealthy_scores} → selected cluster {most_unhealthy}")

        return most_unhealthy

    def _find_most_healthy_cluster(self, X: np.ndarray, cluster_assignments: np.ndarray) -> int:
        """
        Find cluster with most "healthy" characteristics.

        Uses heuristics:
        - Lowest mean noise_level (feature index 3)
        - Highest mean SNR (feature index 4)

        Args:
            X: Feature matrix (n_samples, n_features)
                Expected features: [fwhm, peak_height, peak_area, noise_level, snr, peak_center]
            cluster_assignments: Cluster ID for each sample

        Returns:
            Cluster ID with highest healthy score
        """
        n_clusters = len(self.cluster_labels_)
        healthy_scores = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_samples = X[cluster_mask]

            if len(cluster_samples) == 0:
                continue

            # Feature indices (based on standard feature extraction)
            NOISE_LEVEL_IDX = 3
            SNR_IDX = 4

            # Score based on noise and SNR
            mean_noise = np.mean(cluster_samples[:, NOISE_LEVEL_IDX])
            mean_snr = np.mean(cluster_samples[:, SNR_IDX])

            # Lower noise + higher SNR = more healthy
            healthy_scores[cluster_id] = -mean_noise + 0.1 * mean_snr

        most_healthy = int(np.argmax(healthy_scores))

        logger.debug(f"Healthy scores: {healthy_scores} → selected cluster {most_healthy}")

        return most_healthy

    def predict_cluster_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments and propagated labels for new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (cluster_assignments, propagated_labels)
        """
        if self.kmeans_ is None or self.cluster_labels_ is None:
            raise ValueError("Model not fitted. Call cluster_and_label() first.")

        cluster_assignments = self.kmeans_.predict(X)
        propagated_labels = np.array([self.cluster_labels_[c] for c in cluster_assignments])

        return cluster_assignments, propagated_labels


def select_sliding_window_data(
    df: pd.DataFrame,
    window_size: int | None = None,
    window_days: int | None = None,
    timestamp_col: str = "timestamp",
    label_col: str = "ground_truth_label",
) -> pd.DataFrame:
    """
    Select data from sliding window (most recent N samples or M days).

    Args:
        df: Full dataset
        window_size: Number of most recent samples (if None, use window_days)
        window_days: Number of most recent days (if None, use window_size)
        timestamp_col: Name of timestamp column
        label_col: Name of label column (filters for non-null)

    Returns:
        Filtered DataFrame with sliding window data

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=1000),
        ...     'ground_truth_label': np.random.randint(0, 2, 1000),
        ...     'feature1': np.random.randn(1000),
        ... })
        >>> recent_df = select_sliding_window_data(df, window_days=30)
        >>> print(f"Selected {len(recent_df)} samples from last 30 days")
    """
    # Filter for labeled samples only
    df_labeled = df[df[label_col].notna()].copy()

    if len(df_labeled) == 0:
        raise ValueError("No labeled samples found in dataset")

    # Sort by timestamp
    if timestamp_col in df_labeled.columns:
        df_labeled = df_labeled.sort_values(timestamp_col, ascending=False)

    # Apply sliding window
    if window_size is not None:
        df_windowed = df_labeled.head(window_size)
        logger.info(f"Sliding window: Selected {len(df_windowed)} most recent samples")
    elif window_days is not None:
        if timestamp_col not in df_labeled.columns:
            raise ValueError(f"timestamp_col '{timestamp_col}' not found in DataFrame")

        cutoff_date = df_labeled[timestamp_col].max() - pd.Timedelta(days=window_days)
        df_windowed = df_labeled[df_labeled[timestamp_col] >= cutoff_date]
        logger.info(
            f"Sliding window: Selected {len(df_windowed)} samples from last {window_days} days "
            f"(since {cutoff_date})"
        )
    else:
        raise ValueError("Must specify either window_size or window_days")

    return df_windowed


def create_gold_standard_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 80/20 train/test split with gold standard test set.

    The test set (20%) is the "gold standard" used for Champion/Challenger evaluation.
    It must contain ONLY fully labeled samples (no pseudo-labels).

    Args:
        X: Feature matrix
        y: Label vector (must be fully labeled, no -1 values)
        test_size: Fraction for test set (default 0.2 = 20%)
        stratify: Whether to stratify split by labels
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        >>> X = np.random.randn(1000, 10)
        >>> y = np.random.randint(0, 2, 1000)
        >>> X_train, X_test, y_train, y_test = create_gold_standard_split(X, y)
        >>> print(f"Gold standard test set: {len(y_test)} samples")
    """
    from sklearn.model_selection import train_test_split

    # Verify no unlabeled samples
    if np.any(y == -1):
        raise ValueError(
            "Gold standard split requires fully labeled data. "
            f"Found {np.sum(y == -1)} unlabeled samples."
        )

    stratify_array = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_array,
        random_state=random_state,
    )

    logger.info(
        f"Gold standard split: {len(y_train)} train, {len(y_test)} test (test_size={test_size:.0%})"
    )

    # Log class distribution
    train_healthy = np.sum(y_train == 0)
    train_unhealthy = np.sum(y_train == 1)
    test_healthy = np.sum(y_test == 0)
    test_unhealthy = np.sum(y_test == 1)

    logger.info(
        f"Train set: {train_healthy} healthy, {train_unhealthy} unhealthy "
        f"({100 * train_healthy / len(y_train):.1f}% healthy)"
    )
    logger.info(
        f"Test set (gold standard): {test_healthy} healthy, {test_unhealthy} unhealthy "
        f"({100 * test_healthy / len(y_test):.1f}% healthy)"
    )

    return X_train, X_test, y_train, y_test
