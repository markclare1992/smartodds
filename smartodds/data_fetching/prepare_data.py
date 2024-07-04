from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Any, Dict

import numpy as np
import pandas as pd
from numpy import ndarray, floating
from scipy.optimize import minimize
from scipy.stats import poisson


def create_match_array(df: pd.DataFrame):
    team_1_rates = df['team_1_expected_goals'].values
    team_2_rates = df['team_2_expected_goals'].values
    team_1_ids = df['home_idx'].values
    team_2_ids = df['away_idx'].values
    team_1_conference = df['Conference_home'].values
    team_2_conference = df['Conference_away'].values
    return team_1_ids, team_2_ids, team_1_rates, team_2_rates, team_1_conference, team_2_conference


def create_correct_score_grid(team_1_rate: float, team_2_rate: float,
                              max_score=10) -> ndarray:
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
            correct_score_grid[i, j] = poisson.pmf(i,
                                                   team_1_rate) * poisson.pmf(
                j, team_2_rate
            )
    # normalise
    correct_score_grid /= np.sum(correct_score_grid)
    return correct_score_grid


def create_correct_score_grid_array(team_1_rates: np.ndarray,
                                    team_2_rates: np.ndarray,
                                    max_score=10) -> ndarray:
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
            correct_score_grid[:, i, j] = poisson.pmf(i,
                                                      team_1_rates) * poisson.pmf(
                j, team_2_rates)
    # Normalize
    correct_score_grid /= np.sum(correct_score_grid, axis=(1, 2),
                                 keepdims=True)
    return correct_score_grid


def calculate_win_draw_win_probabilities(correct_score_grid: np.ndarray) -> \
Tuple[
    floating[Any], floating[Any], floating[Any]]:
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
        correct_score_grids: np.ndarray) -> Tuple[
    floating[Any], floating[Any], floating[Any]]:
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


class EvaluationMetrics(Enum):
    LOG_LOSS = 'log_loss'
    BRIER_SCORE = 'brier_score'
    BIAS = 'bias'
    RMSE = 'rmse'


def log_loss(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def brier_score(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.mean((y_true - y_pred) ** 2)


def bias(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.mean(y_pred - y_true)


def rmse(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


metric_to_function = {
    EvaluationMetrics.LOG_LOSS: log_loss,
    EvaluationMetrics.BRIER_SCORE: brier_score,
    EvaluationMetrics.BIAS: bias,
    EvaluationMetrics.RMSE: rmse
}


class BaseModelClass(ABC):
    def __init__(self, data_file_path: str, comparison_file_path: str):
        self.data_file_path = data_file_path
        self.comparison_file_path = comparison_file_path
        self._train, self._test, self._team_to_idx = self._prepare_data()
        self._compare_data = self._prepare_comparison_data()
        self._result = None
        self._attack_rates = None
        self._defense_rates = None
        self._gamma = None
        self._eta = None
        self.position_probabilities_western = None
        self.position_probabilities_eastern = None

    @property
    def train(self) -> pd.DataFrame:
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        return self._test

    @property
    def compare_data(self) -> pd.DataFrame:
        return self._compare_data

    @property
    def team_to_idx(self) -> Dict[str, int]:
        return self._team_to_idx

    @property
    def idx_to_team(self) -> Dict[int, str]:
        return {v: k for k, v in self._team_to_idx.items()}

    @property
    def n_teams(self) -> int:
        return len(self._team_to_idx)

    @property
    def attack_rates(self) -> np.ndarray:
        return self._attack_rates

    @property
    def defense_rates(self) -> np.ndarray:
        return self._defense_rates

    @property
    def gamma(self) -> floating[Any]:
        return self._gamma

    @property
    def eta(self) -> floating[Any]:
        return self._eta

    @property
    def result(self) -> np.ndarray:
        if self._result is None:
            raise ValueError("Model has not been fitted yet.")
        return self._result

    def _prepare_data(self) -> Tuple[
        pd.DataFrame, pd.DataFrame, dict[str, int]]:
        df = pd.read_csv(self.data_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        fitset = df[df['Dataset'] == 'fitset'].copy()
        unique_fitset_teams = sorted(
            pd.concat([fitset['Home'], fitset['Away']]).unique())
        team_to_idx = {team: idx for idx, team in
                       enumerate(unique_fitset_teams)}
        fitset['home_idx'] = fitset['Home'].map(team_to_idx)
        fitset['away_idx'] = fitset['Away'].map(team_to_idx)

        testset = df[df['Dataset'] == 'simulation_set'].copy()
        test_initial_n_matches = len(testset)
        testset = testset[
            testset['Home'].isin(unique_fitset_teams) & testset['Away'].isin(
                unique_fitset_teams)]
        test_final_n_matches = len(testset)
        print(
            f"Removed {test_initial_n_matches - test_final_n_matches} matches from the test set.")
        testset['home_idx'] = testset['Home'].map(team_to_idx)
        testset['away_idx'] = testset['Away'].map(team_to_idx)

        return fitset, testset, team_to_idx

    @abstractmethod
    def _calculate_initial_parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        pass

    @abstractmethod
    def log_likelihood(self, params: np.ndarray) -> float:
        pass

    def fit(self):
        initial_params = self._calculate_initial_parameters()
        bounds = self.bounds()
        result = minimize(self.log_likelihood, initial_params,
                          method='L-BFGS-B', bounds=bounds)
        self._result = result
        self._extract_fitted_params()

    @abstractmethod
    def _extract_fitted_params(self):
        pass

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        pass

    def add_predictions(self):
        if self.result:
            self._test = self.predict(self.test)

    def evaluate(self) -> dict[str, floating[Any]]:
        if not self.result:
            raise ValueError("Model has not been fitted yet.")

        home_win = self.test['Res'] == 'H'
        away_win = self.test['Res'] == 'A'
        draw_win = self.test['Res'] == 'D'

        home_probs = self.test['home_win']
        away_probs = self.test['away_win']
        draw_probs = self.test['draw']

        total_goals = self.test['HG'] + self.test['AG']

        metrics = {}
        for metric in EvaluationMetrics:
            home_metric = metric_to_function[metric](home_win, home_probs)
            away_metric = metric_to_function[metric](away_win, away_probs)
            draw_metric = metric_to_function[metric](draw_win, draw_probs)
            mean_metric = np.mean([home_metric, away_metric, draw_metric])
            metrics[f'home_{metric.value}'] = home_metric
            metrics[f'away_{metric.value}'] = away_metric
            metrics[f'draw_{metric.value}'] = draw_metric
            metrics[f'mean_{metric.value}'] = mean_metric

        metrics['total_goals_rmse'] = rmse(total_goals,
                                           self.test['total_expected_goals'])

        return metrics

    @abstractmethod
    def display_fitted_model_params(self):
        pass

    @staticmethod
    def _calculate_expected_points(win_probability: float,
                                   draw_probability: float) -> float:
        return win_probability * 3 + draw_probability

    def add_expected_points(self):
        """
        Add home expected points and away expected points to the test set.
        """
        self._test['home_expected_points'] = self._calculate_expected_points(
            self._test['home_win'], self._test['draw'])
        self._test['away_expected_points'] = self._calculate_expected_points(
            self._test['away_win'], self._test['draw'])

    def create_expected_points_table(self):
        """
        Create a table of expected points for each team in the test set.
        """
        expected_points_table = self._test.groupby('Home')[
                                    'home_expected_points'].sum() + \
                                self._test.groupby('Away')[
                                    'away_expected_points'].sum()
        # order and add position
        expected_points_table = expected_points_table.sort_values(
            ascending=False)
        expected_points_table = expected_points_table.reset_index()
        return expected_points_table

    def _prepare_comparison_data(self):
        comparison_data = pd.read_csv(self.comparison_file_path)
        comparison_data['Date'] = pd.to_datetime(comparison_data['Date'])
        comparison_data['month'] = comparison_data['Date'].dt.month
        comparison_data['home_idx'] = comparison_data['Home'].map(
            self.team_to_idx)
        comparison_data['away_idx'] = comparison_data['Away'].map(
            self.team_to_idx)

        comparison_data['total_expected_goals'] = comparison_data[
                                                      'expected_team1_goals'] + \
                                                  comparison_data[
                                                      'expected_team2_goals']

        # merge with test data based on home_idx and away_idx, and date
        comparison_data = comparison_data.merge(
            self.test[['home_idx', 'away_idx', 'Date', 'Res', 'HG', 'AG']],
            on=['home_idx', 'away_idx', 'Date'], how='inner')
        return comparison_data

    def evaluate_comparison_data(self):
        home_win = self.compare_data['Res'] == 'H'
        away_win = self.compare_data['Res'] == 'A'
        draw_win = self.compare_data['Res'] == 'D'

        home_probs = self.compare_data['expected_team1_win']
        away_probs = self.compare_data['expected_team2_win']
        draw_probs = self.compare_data['expected_draw']

        total_goals = self.compare_data['HG'] + self.compare_data['AG']

        metrics = {}
        for metric in EvaluationMetrics:
            home_metric = metric_to_function[metric](home_win, home_probs)
            away_metric = metric_to_function[metric](away_win, away_probs)
            draw_metric = metric_to_function[metric](draw_win, draw_probs)
            mean_metric = np.mean([home_metric, away_metric, draw_metric])
            metrics[f'home_{metric.value}'] = home_metric
            metrics[f'away_{metric.value}'] = away_metric
            metrics[f'draw_{metric.value}'] = draw_metric
            metrics[f'mean_{metric.value}'] = mean_metric

        metrics['total_goals_rmse'] = rmse(total_goals,
                                           self.compare_data[
                                               'total_expected_goals'])

        return metrics

    def simulate_season(self, team_1_ids, team_2_ids, team_1_rates,
                        team_2_rates, team_1_conferences, team_2_conferences):
        """
        Simulate an entire season based on the test data and return the match results.

        Parameters:
            team_1_ids (np.ndarray): Array of home team indices.
            team_2_ids (np.ndarray): Array of away team indices.
            team_1_rates (np.ndarray): Array of home team scoring rates.
            team_2_rates (np.ndarray): Array of away team scoring rates.
            team_1_conferences (np.ndarray): Array of home team conferences.
            team_2_conferences (np.ndarray): Array of away team conferences.

        Returns:
            np.ndarray: Array with the simulated results containing home_goals, away_goals, home_idx, and away_idx.
        """
        home_goals = poisson.rvs(team_1_rates)
        away_goals = poisson.rvs(team_2_rates)
        results = np.column_stack(
            (
            team_1_ids, team_2_ids, home_goals, away_goals, team_1_conferences,
            team_2_conferences))
        return results

    def create_league_table(self, season_results: np.ndarray,
                            num_teams: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create league tables for each conference based on the season results.

        Parameters:
            season_results (np.ndarray): Array with the simulated match results containing home_goals, away_goals, home_idx, away_idx, home_conference, away_conference.
            num_teams (int): Number of teams.

        Returns:
            Tuple: Two arrays representing the league tables for each conference with team indices and points.
        """
        points_eastern = np.zeros(num_teams, dtype=int)
        points_western = np.zeros(num_teams, dtype=int)
        for match in season_results:
            home_idx = match[0]
            away_idx = match[1]
            home_goals = match[2]
            away_goals = match[3]
            home_conference = match[4]
            away_conference = match[5]

            if home_goals > away_goals:
                if home_conference == 'Eastern':
                    points_eastern[home_idx] += 3
                elif home_conference == 'Western':
                    points_western[home_idx] += 3
            elif home_goals < away_goals:
                if away_conference == 'Eastern':
                    points_eastern[away_idx] += 3
                elif away_conference == 'Western':
                    points_western[away_idx] += 3
            else:
                if home_conference == 'Eastern':
                    points_eastern[home_idx] += 1
                elif home_conference == 'Western':
                    points_western[home_idx] += 1
                if away_conference == 'Eastern':
                    points_eastern[away_idx] += 1
                elif away_conference == 'Western':
                    points_western[away_idx] += 1

        table_eastern = np.column_stack((np.arange(num_teams), points_eastern))
        table_western = np.column_stack((np.arange(num_teams), points_western))

        table_eastern = table_eastern[np.argsort(
            -table_eastern[:, 1])]  # Sort by points in descending order
        table_western = table_western[np.argsort(
            -table_western[:, 1])]  # Sort by points in descending order

        return table_eastern, table_western

    def run_simulations(self, num_simulations: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Run multiple simulations of the season and store the final positions for each conference.

        Parameters:
            num_simulations (int): Number of simulations to run.

        Returns:
            Tuple: Arrays with the final positions for each simulation in each conference.
        """
        team_1_ids, team_2_ids, team_1_rates, team_2_rates, team_1_conferences, team_2_conferences = create_match_array(
            self.test)
        num_teams = len(self.team_to_idx)
        final_positions_eastern = np.zeros((num_simulations, num_teams),
                                           dtype=int)
        final_positions_western = np.zeros((num_simulations, num_teams),
                                           dtype=int)

        for i in range(num_simulations):
            season_results = self.simulate_season(team_1_ids, team_2_ids,
                                                  team_1_rates, team_2_rates,
                                                  team_1_conferences,
                                                  team_2_conferences)
            league_table_eastern, league_table_western = self.create_league_table(
                season_results, num_teams)
            final_positions_eastern[i] = league_table_eastern[:, 0]
            final_positions_western[i] = league_table_western[:, 0]

        position_counts_eastern = np.zeros((num_teams, num_teams), dtype=int)
        position_counts_western = np.zeros((num_teams, num_teams), dtype=int)

        for i in range(num_simulations):
            for position, team in enumerate(final_positions_eastern[i]):
                position_counts_eastern[team, position] += 1
            for position, team in enumerate(final_positions_western[i]):
                position_counts_western[team, position] += 1

        position_probabilities_eastern = position_counts_eastern / num_simulations
        position_probabilities_western = position_counts_western / num_simulations

        self.position_probabilities_eastern = position_probabilities_eastern
        self.position_probabilities_western = position_probabilities_western

        return position_probabilities_eastern, position_probabilities_western

    def get_team__name_position_probability(self, team: str,
                                            position_range: Tuple[int, int]) -> \
            Tuple[float, float]:
        """
        Get the probability of a team finishing within a given range of positions.

        Parameters:
            team (str): The team name or ID.
            position_range (Tuple[int, int]): The range of positions (inclusive).

        Returns:
            Tuple: Probabilities of finishing in the specified range for Eastern and Western conferences.
        """
        team_id = self.team_to_idx[team]

        start_pos, end_pos = position_range
        eastern_prob = np.sum(self.position_probabilities_eastern[team_id,
                              start_pos - 1:end_pos])
        western_prob = np.sum(self.position_probabilities_western[team_id,
                              start_pos - 1:end_pos])

        return eastern_prob, western_prob


class SimpleModel(BaseModelClass):

    def _calculate_initial_parameters(self) -> np.ndarray:
        # Calculate observed average goals
        observed_avg_goals = self.train['HG'].mean() + self.train['AG'].mean()
        initial_gamma = np.log(observed_avg_goals)

        # Calculate initial attack and defense rates
        attack_initial = np.log(
            self.train.groupby('home_idx')['HG'].mean().reindex(
                range(self.n_teams), fill_value=0).values + 1e-6)
        defense_initial = np.log(
            self.train.groupby('home_idx')['AG'].mean().reindex(
                range(self.n_teams), fill_value=0).values + 1e-6)

        # calculate initial eta
        observed_avg_home_advantage = self.train['HG'].mean() - self.train[
            'AG'].mean()
        initial_eta = np.log(observed_avg_home_advantage + 1e-6)

        # Initial parameter estimates
        initial_params = np.concatenate(
            [attack_initial, defense_initial, [initial_gamma], [initial_eta]])
        return initial_params

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Ensure the number of bounds matches the number of parameters
        bounds = [(None, None)] * 2 * self.n_teams + [(0, None), (None, None)]
        return bounds

    def log_likelihood(self, params: np.ndarray) -> float:
        attack_rates = params[:self.n_teams]
        defense_rates = params[self.n_teams:2 * self.n_teams]
        gamma = params[2 * self.n_teams]
        eta = params[2 * self.n_teams + 1]

        # Sum-to-zero constraints
        attack_rates -= attack_rates.mean()
        defense_rates -= defense_rates.mean()

        home_idx = self.train['home_idx'].values
        away_idx = self.train['away_idx'].values
        home_goals = self.train['HG'].values
        away_goals = self.train['AG'].values

        lambda_k = np.exp(
            attack_rates[home_idx] + defense_rates[away_idx] + gamma + eta / 2)
        mu_k = np.exp(
            attack_rates[away_idx] + defense_rates[home_idx] + gamma - eta / 2)

        log_like = np.sum(np.log(poisson.pmf(home_goals, lambda_k))) + np.sum(
            np.log(poisson.pmf(away_goals, mu_k)))

        return -log_like

    def _extract_fitted_params(self):
        fitted_params = self.result.x
        self._attack_rates = fitted_params[:self.n_teams]
        self._defense_rates = fitted_params[self.n_teams:2 * self.n_teams]
        self._gamma = fitted_params[2 * self.n_teams]
        self._eta = fitted_params[2 * self.n_teams + 1]

        # Sum-to-zero constraints
        self._attack_rates -= self.attack_rates.mean()
        self._defense_rates -= self.defense_rates.mean()

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        home_idx = test_data['home_idx'].values
        away_idx = test_data['away_idx'].values

        lambda_k, mu_k = self._calculate_team_lambdas(away_idx, home_idx)
        test_data['team_1_expected_goals'] = lambda_k
        test_data['team_2_expected_goals'] = mu_k
        test_data['total_expected_goals'] = lambda_k + mu_k

        home_win, draw, away_win = calculate_win_draw_win_probabilities_array(
            create_correct_score_grid_array(lambda_k, mu_k))
        test_data['home_win'] = home_win
        test_data['draw'] = draw
        test_data['away_win'] = away_win

        return test_data

    def _calculate_team_lambdas(self, away_idx: np.ndarray,
                                home_idx: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        lambda_k = np.exp(
            self.attack_rates[home_idx] + self.defense_rates[away_idx] +
            self.gamma + self.eta / 2)
        mu_k = np.exp(
            self.attack_rates[away_idx] + self.defense_rates[home_idx] +
            self.gamma - self.eta / 2)
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


# model = SimpleModel('merged_data.csv','data_mls_simset_predictions.csv')
# model.fit()
# model.display_fitted_model_params()
# model.add_predictions()
# model.evaluate()


class ExtendedModel(BaseModelClass):

    def _calculate_initial_parameters(self) -> np.ndarray:
        # Calculate observed average goals
        observed_avg_goals = self.train['HG'].mean() + self.train['AG'].mean()
        # Repeat 12 times, one for each month
        initial_gamma = np.repeat(np.log(observed_avg_goals), 12)

        # Calculate initial attack and defense rates
        attack_initial = np.log(
            self.train.groupby('home_idx')['HG'].mean().reindex(
                range(self.n_teams), fill_value=0).values + 1e-6)
        defense_initial = np.log(
            self.train.groupby('away_idx')['AG'].mean().reindex(
                range(self.n_teams), fill_value=0).values + 1e-6)

        # Calculate initial eta
        observed_avg_home_advantage = self.train['HG'].mean() - self.train[
            'AG'].mean()
        initial_eta = np.log(observed_avg_home_advantage + 1e-6)

        # Initial parameter estimates
        initial_params = np.concatenate(
            [attack_initial, defense_initial, initial_gamma, [initial_eta]])
        return initial_params

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Ensure the number of bounds matches the number of parameters
        bounds = [(None, None)] * (2 * self.n_teams) + [(None, None)] * 12 + [
            (0, None)]
        return bounds

    def log_likelihood(self, params: np.ndarray) -> float:
        attack_rates = params[:self.n_teams]
        defense_rates = params[self.n_teams:2 * self.n_teams]
        monthly_gamma = params[2 * self.n_teams: 2 * self.n_teams + 12]
        eta = params[2 * self.n_teams + 12]

        # Sum-to-zero constraints
        attack_rates -= attack_rates.mean()
        defense_rates -= defense_rates.mean()

        home_idx = self.train['home_idx'].values
        away_idx = self.train['away_idx'].values
        home_goals = self.train['HG'].values
        away_goals = self.train['AG'].values

        month = self.train[
                    'month'].values - 1  # Adjusting month to be zero-indexed

        lambda_k = np.exp(
            attack_rates[home_idx] + defense_rates[away_idx] + monthly_gamma[
                month] + eta
        )
        mu_k = np.exp(
            attack_rates[away_idx] + defense_rates[home_idx] + monthly_gamma[
                month] - eta
        )

        log_like = np.sum(np.log(poisson.pmf(home_goals, lambda_k))) + np.sum(
            np.log(poisson.pmf(away_goals, mu_k)))

        return -log_like

    def _extract_fitted_params(self):
        fitted_params = self.result.x
        self._attack_rates = fitted_params[:self.n_teams]
        self._defense_rates = fitted_params[self.n_teams:2 * self.n_teams]
        self._gamma = fitted_params[2 * self.n_teams: 2 * self.n_teams + 12]
        self._eta = fitted_params[2 * self.n_teams + 12]

        # Sum-to-zero constraints
        self._attack_rates -= self.attack_rates.mean()
        self._defense_rates -= self.defense_rates.mean()

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        home_idx = test_data['home_idx'].values
        away_idx = test_data['away_idx'].values
        month = test_data['month'].values - 1

        lambda_k, mu_k = self._calculate_team_lambdas(away_idx, home_idx,
                                                      month)
        test_data['team_1_expected_goals'] = lambda_k
        test_data['team_2_expected_goals'] = mu_k
        test_data['total_expected_goals'] = lambda_k + mu_k
        home_win, draw, away_win = calculate_win_draw_win_probabilities_array(
            create_correct_score_grid_array(lambda_k, mu_k))
        test_data['home_win'] = home_win
        test_data['draw'] = draw
        test_data['away_win'] = away_win

        return test_data

    def _calculate_team_lambdas(self, away_idx: np.ndarray,
                                home_idx: np.ndarray, month: np.ndarray) -> \
    Tuple[
        np.ndarray, np.ndarray]:
        lambda_k = np.exp(
            self.attack_rates[home_idx] + self.defense_rates[away_idx] +
            self.gamma[month] + self.eta
        )
        mu_k = np.exp(
            self.attack_rates[away_idx] + self.defense_rates[home_idx] +
            self.gamma[month] - self.eta
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

# extended_model = ExtendedModel('merged_data.csv','data_mls_simset_predictions.csv')
# extended_model.fit()
# extended_model.display_fitted_model_params()
# extended_model.add_predictions()
# eval=extended_model.evaluate()
# comparison_eval = extended_model.evaluate_comparison_data()
# extended_model.add_expected_points()
# table = extended_model.create_expected_points_table()
# final_positions = extended_model.run_simulations(10**4)
# team_name = 'Los Angeles Galaxy'
# position_range = (1,2)
# eastern_prob, western_prob = extended_model.get_team__name_position_probability(team_name, position_range)
