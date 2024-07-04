"""Simple Poisson model for football match prediction."""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from smartodds.model.base_model import BaseModelClass
from smartodds.model.utils import (
    calculate_win_draw_win_probabilities_array,
    create_correct_score_grid_array,
)


class SimpleModel(BaseModelClass):
    def _calculate_initial_parameters(self) -> np.ndarray:
        # Calculate observed average goals
        observed_avg_goals = self.train["HG"].mean() + self.train["AG"].mean()
        initial_gamma = np.log(observed_avg_goals)

        # Calculate initial attack and defense rates
        attack_initial = np.log(
            self.train.groupby("home_idx")["HG"]
            .mean()
            .reindex(range(self.n_teams), fill_value=0)
            .values
            + 1e-6
        )
        defense_initial = np.log(
            self.train.groupby("home_idx")["AG"]
            .mean()
            .reindex(range(self.n_teams), fill_value=0)
            .values
            + 1e-6
        )

        # calculate initial eta
        observed_avg_home_advantage = self.train["HG"].mean() - self.train["AG"].mean()
        initial_eta = np.log(observed_avg_home_advantage + 1e-6)

        # Initial parameter estimates
        initial_params = np.concatenate(
            [attack_initial, defense_initial, [initial_gamma], [initial_eta]]
        )
        return initial_params

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Ensure the number of bounds matches the number of parameters
        bounds = [(None, None)] * 2 * self.n_teams + [(0, None), (None, None)]
        return bounds

    def log_likelihood(self, params: np.ndarray) -> float:
        attack_rates = params[: self.n_teams]
        defense_rates = params[self.n_teams : 2 * self.n_teams]
        gamma = params[2 * self.n_teams]
        eta = params[2 * self.n_teams + 1]

        # Sum-to-zero constraints
        attack_rates -= attack_rates.mean()
        defense_rates -= defense_rates.mean()

        home_idx = self.train["home_idx"].values
        away_idx = self.train["away_idx"].values
        home_goals = self.train["HG"].values
        away_goals = self.train["AG"].values

        lambda_k = np.exp(
            attack_rates[home_idx] + defense_rates[away_idx] + gamma + eta / 2
        )
        mu_k = np.exp(
            attack_rates[away_idx] + defense_rates[home_idx] + gamma - eta / 2
        )

        log_like = np.sum(np.log(poisson.pmf(home_goals, lambda_k))) + np.sum(
            np.log(poisson.pmf(away_goals, mu_k))
        )

        return -log_like

    def _extract_fitted_params(self):
        fitted_params = self.result.x
        self._attack_rates = fitted_params[: self.n_teams]
        self._defense_rates = fitted_params[self.n_teams : 2 * self.n_teams]
        self._gamma = fitted_params[2 * self.n_teams]
        self._eta = fitted_params[2 * self.n_teams + 1]

        # Sum-to-zero constraints
        self._attack_rates -= self.attack_rates.mean()
        self._defense_rates -= self.defense_rates.mean()

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        home_idx = test_data["home_idx"].values
        away_idx = test_data["away_idx"].values

        lambda_k, mu_k = self._calculate_team_lambdas(away_idx, home_idx)
        test_data["team_1_expected_goals"] = lambda_k
        test_data["team_2_expected_goals"] = mu_k
        test_data["total_expected_goals"] = lambda_k + mu_k

        home_win, draw, away_win = calculate_win_draw_win_probabilities_array(
            create_correct_score_grid_array(lambda_k, mu_k)
        )
        test_data["home_win"] = home_win
        test_data["draw"] = draw
        test_data["away_win"] = away_win

        return test_data

    def _calculate_team_lambdas(
        self, away_idx: np.ndarray, home_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lambda_k = np.exp(
            self.attack_rates[home_idx]
            + self.defense_rates[away_idx]
            + self.gamma
            + self.eta / 2
        )
        mu_k = np.exp(
            self.attack_rates[away_idx]
            + self.defense_rates[home_idx]
            + self.gamma
            - self.eta / 2
        )
        return lambda_k, mu_k

    def display_fitted_model_params(self):
        if not self.result:
            raise ValueError("Model has not been fitted yet.")
        print("Optimization result:")
        print(self.result)
        print(f"Attack rates: {self.attack_rates}")
        print(f"Defense rates: {self.defense_rates}")
        print(f"Gamma: {self.gamma}")
        print(f"Eta: {self.eta}")
