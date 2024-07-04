"""Utility functions for the model module."""
from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy import floating, ndarray
from scipy.stats import poisson


def create_match_array(df: pd.DataFrame):
    team_1_rates = df["team_1_expected_goals"].values
    team_2_rates = df["team_2_expected_goals"].values
    team_1_ids = df["home_idx"].values
    team_2_ids = df["away_idx"].values
    team_1_conference = df["Conference_home"].values
    team_2_conference = df["Conference_away"].values
    return (
        team_1_ids,
        team_2_ids,
        team_1_rates,
        team_2_rates,
        team_1_conference,
        team_2_conference,
    )


def create_correct_score_grid(
    team_1_rate: float, team_2_rate: float, max_score=10
) -> ndarray:
    """
    Assume goalscoring rates are independent poison processes.
    Create probability matrix for each possible score.

    Parameters:
        team_1_rate (float): The goalscoring rate for team 1.
        team_2_rate (float): The goalscoring rate for team 2.
        max_score (int): The maximum score to consider.

    Returns:
        np.ndarray: The normalised probability matrix for each possible score.
    """
    correct_score_grid = np.zeros((max_score + 1, max_score + 1))
    for i in range(max_score + 1):
        for j in range(max_score + 1):
            correct_score_grid[i, j] = poisson.pmf(i, team_1_rate) * poisson.pmf(
                j, team_2_rate
            )
    # normalise
    correct_score_grid /= np.sum(correct_score_grid)
    return correct_score_grid


def create_correct_score_grid_array(
    team_1_rates: np.ndarray, team_2_rates: np.ndarray, max_score=10
) -> ndarray:
    """
    Create probability matrices for each possible score across multiple matches.
    Assume goalscoring rates are independent Poisson processes.

    Parameters:
        team_1_rates (np.ndarray): An array of goalscoring rates for team 1.
        team_2_rates (np.ndarray): An array of goalscoring rates for team 2.
        max_score (int): The maximum score to consider (default is 10).

    Returns:
        np.ndarray: The normalized probability matrices for each possible score.
    """
    num_matches = team_1_rates.size
    correct_score_grid = np.zeros((num_matches, max_score + 1, max_score + 1))
    for i in range(max_score + 1):
        for j in range(max_score + 1):
            correct_score_grid[:, i, j] = poisson.pmf(i, team_1_rates) * poisson.pmf(
                j, team_2_rates
            )
    # Normalize
    correct_score_grid /= np.sum(correct_score_grid, axis=(1, 2), keepdims=True)
    return correct_score_grid


def calculate_win_draw_win_probabilities(
    correct_score_grid: np.ndarray,
) -> Tuple[floating[Any], floating[Any], floating[Any]]:
    """
    Calculate the win-draw-win probabilities from the correct score grid.

    Parameters:
        correct_score_grid (np.ndarray): The normalised probability matrix for each possible score.

    Returns:
        tuple: The win-draw-win probabilities.
    """
    home_win = np.sum(np.tril(correct_score_grid, -1))
    draw = np.sum(np.diag(correct_score_grid))
    away_win = np.sum(np.triu(correct_score_grid, 1))
    return home_win, draw, away_win


def calculate_win_draw_win_probabilities_array(
    correct_score_grids: np.ndarray,
) -> Tuple[floating[Any], floating[Any], floating[Any]]:
    """
    Calculate the win-draw-win probabilities from the correct score grids for multiple matches.

    Parameters:
        correct_score_grids (np.ndarray): The normalized probability matrices for each possible score.

    Returns:
        tuple: The win-draw-win probabilities (home win, draw, away win) for each match.
    """
    home_win = np.sum(np.tril(correct_score_grids, -1), axis=(1, 2))
    draw = np.sum(np.diagonal(correct_score_grids, axis1=1, axis2=2), axis=1)
    away_win = np.sum(np.triu(correct_score_grids, 1), axis=(1, 2))
    return home_win, draw, away_win
