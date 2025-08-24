"""Geometric metrics for SPD component similarity and common dimension projection.

This implementation is directly adapted from saman/Common_space (2).ipynb
to integrate geometric metrics into the main SPD training loop.
"""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.models.components import EmbeddingComponent, LinearComponent

EPS = 1e-12


def effect_vector_features_single_component(
    A_vector: Float[Tensor, "d_in"],  # Single column from A matrix
    B_vector: Float[Tensor, "d_out"],  # Single row from B matrix
    layer_name: str,
    W: Float[Tensor, "d_hidden d_in"] | None,  # PyTorch Linear weight shape
    WT: Float[Tensor, "d_out d_hidden"] | None,  # PyTorch Linear weight shape
    I: Float[Tensor, "d_hidden d_hidden"] | None = None,
) -> Float[Tensor, "d_out"] | None:
    """
    Compute output-space effect vector for a single component (unit-norm vector).
    Directly adapted from notebook's effect_vector_features function.
    
    For TMS models:
    - linear1 (W): v = WT @ (I @ (W @ U))  where U is component's column vector
    - identity (I): v = WT @ U
    - linear2 (WT): v = V  where V is component's row vector
    
    Args:
        A_vector: Single component's input projection vector (U in notebook)
        B_vector: Single component's output projection vector (V in notebook)
        layer_name: Name of the layer
        W: First layer weights from linear1.weight (d_hidden, d_in)
        WT: Last layer weights from linear2.weight (d_out, d_hidden)
        I: Identity layer weights if present (d_hidden, d_hidden)
        
    Returns:
        Unit-norm effect vector or None if computation fails
    """
    # Determine layer type (matching notebook's _layer_kind logic)
    if "linear1" in layer_name:
        # First layer: effect through full network
        if W is None or WT is None:
            return None
        # U is the A_vector (column from A matrix)
        if I is not None:
            v = torch.matmul(WT, torch.matmul(I, torch.matmul(W, A_vector)))
        else:
            # Use identity if no hidden layer
            I_default = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            v = torch.matmul(WT, torch.matmul(I_default, torch.matmul(W, A_vector)))
    elif "identity" in layer_name or "hidden_layers" in layer_name:
        # Identity/hidden layer
        if WT is None:
            return None
        v = torch.matmul(WT, A_vector)
    elif "linear2" in layer_name:
        # Output layer: direct effect
        v = B_vector
    else:
        # Fallback: try V then U if shapes fit (from notebook)
        if B_vector is not None:
            v = B_vector
        elif A_vector is not None:
            v = A_vector
        else:
            return None
    
    # Normalize to unit vector
    v = v.reshape(-1)
    norm = torch.norm(v)
    if norm < EPS:
        return None
    return v / norm


def jl_target_dim(n_points: int, eps: float = 0.2) -> int:
    """
    Calculate target dimension for Johnson-Lindenstrauss projection.
    From notebook: m ≈ 4 ln(n) / (eps^2/2 − eps^3/3)
    
    Args:
        n_points: Number of points to project
        eps: Distortion parameter (smaller = less distortion, higher dimension)
        
    Returns:
        Target dimension m
    """
    if n_points <= 1:
        return 2
    denom = (eps**2 / 2.0) - (eps**3 / 3.0)
    if denom <= 0:
        denom = eps**2 / 2.0
    m = int(math.ceil(4.0 * math.log(n_points) / denom))
    return max(2, m)


def jl_project(
    X: Float[Tensor, "n d"],
    m: int,
    seed: int = 0,
) -> Float[Tensor, "n m"]:
    """
    Johnson-Lindenstrauss random projection to lower dimension.
    From notebook's jl_project function.
    
    Projects n points in d dimensions to m dimensions while approximately
    preserving pairwise distances.
    
    Args:
        X: Input matrix (n points, d dimensions)
        m: Target dimension
        seed: Random seed for reproducibility
        
    Returns:
        Projected matrix (n points, m dimensions), row-normalized
    """
    n, d = X.shape
    device = X.device
    dtype = X.dtype
    
    # Set seed for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # Random Gaussian projection matrix
    R = torch.randn(d, m, device=device, dtype=dtype, generator=generator)
    
    # Project and scale (from notebook)
    Y = torch.matmul(X, R / math.sqrt(m))
    
    # Normalize rows to unit length
    norms = torch.norm(Y, dim=1, keepdim=True) + EPS
    return Y / norms


def build_common_effect_matrix(
    components: Dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    jl_dim: Optional[int] = None,
    jl_seed: int = 0,
    device: str = "cpu",
) -> Float[Tensor, "n_total_components output_dim"]:
    """
    Build matrix of effect vectors for all components in common output space.
    Adapted from notebook's build_common_effect_matrix.
    
    Args:
        components: Dictionary of SPD components by layer name
        target_model: Original model to extract weight matrices from
        jl_dim: Optional JL projection dimension (None = no projection)
        jl_seed: Random seed for JL projection
        device: Device to compute on
        
    Returns:
        Matrix where each row is a component's unit-norm effect vector
    """
    # Extract weight matrices from target model for TMS
    W = None  # First layer weights
    WT = None  # Last layer weights
    I = None  # Identity layer weights (if present)
    
    # Extract TMS model weights - notebook line 867-869 shows these are used directly
    # W = state["linear1.weight"].cpu()  # (hidden, features) 
    # WT = state["linear2.weight"].cpu()  # (features, hidden)
    if hasattr(target_model, "linear1"):
        W = target_model.linear1.weight.to(device)  # Shape: (d_hidden, d_in)
    if hasattr(target_model, "linear2"):
        WT = target_model.linear2.weight.to(device)  # Shape: (d_out, d_hidden)
    if hasattr(target_model, "hidden_layers") and target_model.hidden_layers is not None:
        # Extract identity layer if present
        if len(target_model.hidden_layers) > 0:
            I = target_model.hidden_layers[0].weight.to(device)
    
    # Build effect vectors for each individual component
    effect_vectors = []
    
    for layer_name, component in components.items():
        component = component.to(device)
        
        # Process each of the C components individually (like notebook)
        for c_idx in range(component.C):
            # Extract single component vectors
            A_vector = component.A[:, c_idx]  # d_in dimensional vector (U in notebook)
            B_vector = component.B[c_idx, :]  # d_out dimensional vector (V in notebook)
            
            # Compute effect vector for this single component
            v = effect_vector_features_single_component(
                A_vector=A_vector,
                B_vector=B_vector,
                layer_name=layer_name,
                W=W,
                WT=WT,
                I=I,
            )
            
            if v is None:
                # Use zero vector if computation fails (from notebook)
                output_dim = WT.shape[1] if WT is not None else W.shape[1] if W is not None else 100
                v = torch.zeros(output_dim, device=device, dtype=component.A.dtype)
            
            effect_vectors.append(v)
    
    # Stack into matrix
    V = torch.stack(effect_vectors)
    
    # Apply JL projection if requested (from notebook line 2848-2849)
    if jl_dim is not None:
        V = jl_project(V, jl_dim, seed=jl_seed)
    
    return V


def cosine_similarity_matrix(
    V: Float[Tensor, "n d"],
) -> Float[Tensor, "n n"]:
    """
    Compute cosine similarity matrix from row vectors.
    From notebook's cosine_similarity_matrix.
    
    Args:
        V: Matrix where each row is a vector
        
    Returns:
        Symmetric matrix of cosine similarities
    """
    # Normalize rows
    norms = torch.norm(V, dim=1, keepdim=True) + EPS
    Vn = V / norms
    
    # Compute cosine similarities
    S = torch.matmul(Vn, Vn.T)
    
    return S


def geometric_similarity_loss(
    components: Dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    causal_importances: Dict[str, Float[Tensor, "batch C"]],
    use_jl_projection: bool = True,
    jl_eps: float = 0.2,
    jl_seed: int = 123,
    device: str = "cpu",
) -> Float[Tensor, ""]:
    """
    Compute geometric similarity loss to encourage distinct components.
    
    This loss encourages components with high causal importance to have
    distinct geometric directions in the common effect space.
    
    Args:
        components: Dictionary of SPD components
        target_model: Original model for weight extraction
        causal_importances: Importance scores for weighting
        use_jl_projection: Whether to use JL projection for efficiency
        jl_eps: JL distortion parameter
        jl_seed: Random seed for JL projection
        device: Device to compute on
        
    Returns:
        Scalar loss value
    """
    # Calculate total number of individual components
    n_components = sum(comp.C for comp in components.values())
    
    # Determine JL projection dimension if needed
    jl_dim = None
    if use_jl_projection:
        jl_dim = jl_target_dim(n_components, eps=jl_eps)
    
    # Build common effect matrix (one row per individual component)
    V = build_common_effect_matrix(
        components=components,
        target_model=target_model,
        jl_dim=jl_dim,
        jl_seed=jl_seed,
        device=device,
    )
    
    # Compute cosine similarity matrix
    S = cosine_similarity_matrix(V)
    
    # Weight similarities by importance of component pairs
    # Higher importance components should be more distinct (lower similarity)
    importance_weights = []
    for layer_name, component in components.items():
        # Average importance across batch dimension
        layer_importance = causal_importances[layer_name].mean(dim=0)  # Shape: (C,)
        importance_weights.append(layer_importance)
    
    # Concatenate all importance weights
    importance_vector = torch.cat(importance_weights)  # Shape: (total_components,)
    
    # Create pairwise importance matrix (product of importances)
    importance_matrix = torch.outer(importance_vector, importance_vector)
    
    # Mask diagonal (self-similarity)
    mask = 1.0 - torch.eye(n_components, device=device)
    
    # Loss: penalize high similarity between important components
    # We want important components to be orthogonal (similarity = 0)
    weighted_similarities = S * importance_matrix * mask
    
    # Use squared similarity to penalize both positive and negative correlations
    loss = (weighted_similarities ** 2).sum() / (n_components * (n_components - 1))
    
    return loss