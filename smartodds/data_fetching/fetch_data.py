"""Functionality for fetching data from the football-data.co.uk website."""

import datetime
import time

import numpy as np
import pandas as pd
import requests as req
from bs4 import BeautifulSoup

from smartodds.data_fetching.config import (
    CUSTOM_HEADER,
    FITSET_CUTTOFF,
    RAW_MLS_FOOTBALLDATA_URL,
    SIMULATION_SET_END,
    SIMULATION_SET_START,
    history_url,
)
from smartodds.data_fetching.team_mappings import TEAM_MAPPINGS


def fetch_footballdata_data() -> pd.DataFrame:
    """
    Fetch the football data from the football-data.co.uk website & convert the date column to datetime.

    Returns:
        pd.DataFrame: The football data.
    """
    df = pd.read_csv(RAW_MLS_FOOTBALLDATA_URL)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df


def get_all_valid_seasons() -> dict:
    """
    Get all valid seasons for the MLS from the history page.

    Returns:
        dict: A dictionary of season names and their corresponding links.
    """
    time.sleep(10)
    r = req.get(history_url, headers=CUSTOM_HEADER)
    soup = BeautifulSoup(r.content, "html.parser")
    season_urls = dict(
        [
            (x.text, x.find("a")["href"])
            for x in soup.find_all("th", {"data-stat": True, "class": True})
            if x.find("a") is not None
        ]
    )

    return season_urls


def get_season_link(season: str, season_dict: dict) -> str:
    """Get the link for the given season."""
    return "https://fbref.com/" + season_dict[season]


def find_table_by_caption(tables: list, caption_text: str) -> BeautifulSoup | None:
    """Find a table by its caption text."""
    for table in tables:
        caption = table.find("caption")
        if caption and caption_text in caption.get_text():
            return table
    return None


def get_season_teams(season_link: str, season: str) -> pd.DataFrame | None:
    """
    Get the teams and their conferences for a given season.

    Parameters:
        season_link (str): The link to the season.
        season (str): The season.

    Returns:
        pd.DataFrame | None: The teams and their conferences for the given season.
    """
    time.sleep(10)
    r = req.get(season_link, headers=CUSTOM_HEADER)
    soup = BeautifulSoup(r.content, "html.parser")
    tables = soup.find_all(
        "table", class_="stats_table sortable min_width force_mobilize"
    )

    eastern_table = find_table_by_caption(tables, "Eastern Conference")
    western_table = find_table_by_caption(tables, "Western Conference")

    eastern_df = pd.read_html(str(eastern_table))[0] if eastern_table else None
    western_df = pd.read_html(str(western_table))[0] if western_table else None

    if eastern_df is None or western_df is None:
        return None

    eastern_df["Conference"] = "Eastern"
    western_df["Conference"] = "Western"
    eastern_df["Season"] = int(season)
    western_df["Season"] = int(season)

    eastern_df = eastern_df[["Season", "Squad", "Conference"]]
    western_df = western_df[["Season", "Squad", "Conference"]]

    return pd.concat([eastern_df, western_df])


def get_team_conference_for_seasons(seasons: list, season_dict: dict) -> pd.DataFrame:
    """
    Fetch the league table, split into eastern and western conferences, for a given list of seasons and league.
    Get team names and conference for each team.

    Parameters:
        seasons (list): The list of seasons.
        season_dict (dict): The dictionary of seasons and their corresponding links.

    Returns:
        pd.DataFrame: The teams and their conferences for the given seasons.
    """
    team_conference = pd.DataFrame()
    for season in seasons:
        team_conference = pd.concat(
            [
                team_conference,
                get_season_teams(get_season_link(str(season), season_dict), season),
            ]
        )
    return team_conference


def get_conference_for_seasons(
    starting_season: int, ending_season: int, season_dict: dict
) -> pd.DataFrame:
    """
    Get the conference for each team for a given range of seasons.

    Parameters:
        starting_season (int): The starting season.
        ending_season (int): The ending season.
        season_dict (dict): The dictionary of seasons and their corresponding links.

    Returns:
        pd.DataFrame: The teams and their conferences for the given seasons.
    """
    return get_team_conference_for_seasons(
        [str(i) for i in range(starting_season, ending_season + 1)], season_dict
    )


def apply_mappings(
    mls_conference_per_season_df: pd.DataFrame, football_data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply the mappings to the football data.

    Parameters:
        mls_conference_per_season_df (pd.DataFrame): The MLS conference per season data.
        football_data_df (pd.DataFrame): The football data.

    Returns:
        pd.DataFrame: The football data with the mappings applied.
    """
    football_data_df = football_data_df.merge(
        mls_conference_per_season_df,
        left_on=["Home", "Season"],
        right_on=["Squad", "Season"],
        how="left",
    )
    football_data_df = football_data_df.merge(
        mls_conference_per_season_df,
        left_on=["Away", "Season"],
        right_on=["Squad", "Season"],
        how="left",
        suffixes=("_home", "_away"),
    )
    football_data_df = football_data_df.drop(["Squad_home", "Squad_away"], axis=1)
    return football_data_df


def generate_team_descriptions(mls_team_conference: pd.DataFrame) -> dict:
    """Generate descriptions for each team based on their conference participation across seasons."""
    descriptions = {}

    for team, group in mls_team_conference.groupby("Squad"):
        desc = []
        for conference, seasons in group.groupby("Conference")["Season"]:
            seasons = sorted(seasons.astype(int))
            ranges = []
            start = prev = seasons[0]
            for season in seasons[1:]:
                if season != prev + 1:
                    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                    start = season
                prev = season
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            desc.append(f"{conference} Conference ({', '.join(ranges)})")
        descriptions[team] = f"{team}: {', '.join(desc)}"

    return descriptions


def apply_dataset_flag(
    merged_data: pd.DataFrame,
    training_end: datetime.datetime,
    simulation_start: datetime.datetime,
    simulation_end: datetime.datetime,
    training_start: datetime.datetime | None = None,
) -> pd.DataFrame:
    """
    Apply the dataset flag to the merged data.

    Parameters:
        merged_data (pd.DataFrame): The merged data.
        training_end (datetime.datetime): The training end date.
        simulation_start (datetime.datetime): The simulation start date.
        simulation_end (datetime.datetime): The simulation end date.
        training_start (datetime.datetime): The training start date. Defaults to None.

    Returns:
        pd.DataFrame: The merged data with the dataset flag applied.

    Note:
        If no training start date is provided, it will be set to the minimum date in the data.
    """
    merged_data["Dataset"] = None
    if training_start is None:
        training_start = merged_data["Date"].min()
    merged_data.loc[
        (merged_data["Date"] >= training_start) & (merged_data["Date"] <= training_end),
        "Dataset",
    ] = "fitset"
    merged_data.loc[
        (merged_data["Date"] >= simulation_start)
        & (merged_data["Date"] <= simulation_end),
        "Dataset",
    ] = "simulation_set"
    return merged_data


def _add_extra_details(df: pd.DataFrame) -> pd.DataFrame:
    df["total_goals"] = df["HG"] + df["AG"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    df["home_game_number"] = df.groupby(["Season", "Home"])["Date"].rank()
    df["away_game_number"] = df.groupby(["Season", "Away"])["Date"].rank()
    return df


def get_fixture_conference_summary_date(merged_dataframe: pd.DataFrame):
    """
    Take the results dataframe and filter for fitset, group by home_conference = away_conference, summarise.

    Parameters:
        merged_dataframe (pd.DataFrame): The merged dataframe.

    Returns:
        pd.DataFrame: The fixture conference summary.
    """
    fixture_conference_summary = (
        merged_dataframe.assign(
            same_conference=merged_dataframe["Conference_home"]
            == merged_dataframe["Conference_away"]
        )
        .groupby(["Dataset", "same_conference"])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={True: "same_conference_count", False: "different_conference_count"}
        )
        .assign(
            total=lambda df: df["same_conference_count"]
            + df["different_conference_count"],
            true_percentage=lambda df: df["same_conference_count"] / df["total"] * 100,
            false_percentage=lambda df: df["different_conference_count"]
            / df["total"]
            * 100,
        )
        .reset_index()
    )

    return fixture_conference_summary[
        [
            "Dataset",
            "same_conference_count",
            "different_conference_count",
            "true_percentage",
            "false_percentage",
        ]
    ]


def filter_for_same_conference(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the merged DataFrame for matches where the home and away teams are in the same conference.

    Parameters:
        merged_df (pd.DataFrame): The merged DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    return (
        merged_df.loc[merged_df["Conference_home"] == merged_df["Conference_away"]]
        .assign(Conference=lambda df: df["Conference_home"])
        .assign(goal_diff=lambda df: df["HG"] - df["AG"])
    )


def get_conference_home_advantage_info(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the results dataframe, filter for same conference, group by home_conference,
    calculate mean home goals, mean away goals, total number of fixtures in each group,
    and perform a bootstrapping to determine statistical significance of home advantage.

    Parameters:
        merged_df (pd.DataFrame): The merged dataframe.

    Returns:
        pd.DataFrame: The conference home advantage information with statistical significance.
    """
    conference_home_advantage_info = (
        merged_df.loc[merged_df["Conference_home"] == merged_df["Conference_away"]]
        .groupby("Conference_home")
        .agg(
            mean_home_goals=("HG", "mean"),
            mean_away_goals=("AG", "mean"),
            total_fixtures=("HG", "count"),
        )
        .reset_index()
        .rename(columns={"Conference_home": "Conference"})
        .assign(
            mean_home_advantage=lambda df: df["mean_home_goals"] - df["mean_away_goals"]
        )
    )

    return conference_home_advantage_info


def get_conference_home_advantage_info_with_confidence_intervals(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """ """
    same_conference_df = merged_df.loc[
        merged_df["Conference_home"] == merged_df["Conference_away"]
    ]

    # Group by home conference and calculate mean goals, total fixtures, and home advantage
    conference_home_advantage_info = (
        same_conference_df.groupby("Conference_home")
        .agg(
            mean_home_goals=("HG", "mean"),
            sd_home_goals=("HG", "std"),
            mean_away_goals=("AG", "mean"),
            sd_away_goals=("AG", "std"),
            total_fixtures=("HG", "count"),
        )
        .reset_index()
        .rename(columns={"Conference_home": "Conference"})
        .assign(home_advantage=lambda df: df["mean_home_goals"] - df["mean_away_goals"])
    )
    # Perform bootstrapping for each conference and add confidence intervals to the DataFrame
    ci_bounds = conference_home_advantage_info["Conference"].apply(
        lambda conference: perform_bootstrap_confidence_interval(
            same_conference_df, conference
        )
    )

    (
        conference_home_advantage_info["ci_lower"],
        conference_home_advantage_info["ci_upper"],
    ) = zip(*ci_bounds)

    return conference_home_advantage_info


def bootstrap_mean_diff(home_goals, away_goals, num_samples=10000):
    """
    Perform bootstrapping to estimate the confidence interval for the difference in means.

    Parameters:
        home_goals (pd.Series): Series of home goals.
        away_goals (pd.Series): Series of away goals.
        num_samples (int): Number of bootstrap samples to generate.

    Returns:
        tuple: Lower and upper bounds of the 95% confidence interval for the difference in means.
    """
    diff_means = []
    n = len(home_goals)
    for _ in range(num_samples):
        sample_indices = np.random.choice(n, n, replace=True)
        sample_home_goals = home_goals.iloc[sample_indices].mean()
        sample_away_goals = away_goals.iloc[sample_indices].mean()
        diff_means.append(sample_home_goals - sample_away_goals)

    lower_bound = np.percentile(diff_means, 2.5)
    upper_bound = np.percentile(diff_means, 97.5)
    return lower_bound, upper_bound


def perform_bootstrap_confidence_interval(
    same_conference_df: pd.DataFrame, conference: str, num_samples=10000
) -> tuple:
    """
    Perform bootstrapping to estimate the confidence interval for the difference in means.

    Parameters:
        same_conference_df (pd.DataFrame): DataFrame filtered for matches where the home and away teams are in the same conference.
        conference (str): The conference to perform the bootstrapping on.
        num_samples (int): Number of bootstrap samples to generate.

    Returns:
        tuple: Lower and upper bounds of the 95% confidence interval for the difference in means.
    """
    home_goals = same_conference_df.loc[
        same_conference_df["Conference_home"] == conference, "HG"
    ]
    away_goals = same_conference_df.loc[
        same_conference_df["Conference_home"] == conference, "AG"
    ]
    return bootstrap_mean_diff(home_goals, away_goals, num_samples)


def compare_conference_home_advantage(
    conference_home_advantage_info: pd.DataFrame,
) -> str:
    """
    Compare the home advantage between Eastern and Western Conferences using confidence intervals.

    Parameters:
        conference_home_advantage_info (pd.DataFrame): The DataFrame containing home advantage information and confidence intervals.

    Returns:
        str: A statement indicating whether the home advantage is statistically different between the conferences.
    """
    east_ci = conference_home_advantage_info.loc[
        conference_home_advantage_info["Conference"] == "Eastern",
        ["ci_lower", "ci_upper"],
    ].values[0]
    west_ci = conference_home_advantage_info.loc[
        conference_home_advantage_info["Conference"] == "Western",
        ["ci_lower", "ci_upper"],
    ].values[0]

    if east_ci[1] < west_ci[0] or west_ci[1] < east_ci[0]:
        return "The home advantage is statistically different between the Eastern and Western Conferences."
    else:
        return "The home advantage is not statistically different between the Eastern and Western Conferences."


if __name__ == "__main__":
    valid_seasons = get_all_valid_seasons()
    mls_team_conference = get_conference_for_seasons(2012, 2024, valid_seasons)
    mls_team_conference["Squad"] = mls_team_conference["Squad"].replace(TEAM_MAPPINGS)
    generate_team_descriptions(mls_team_conference)
    mls_team_conference.to_csv("mls_team_conference.csv", index=False)
    football_data = fetch_footballdata_data()
    mls_team_conference = pd.read_csv("mls_team_conference.csv")
    merged_data = apply_mappings(mls_team_conference, football_data)
    merged_data = apply_dataset_flag(
        merged_data, FITSET_CUTTOFF, SIMULATION_SET_START, SIMULATION_SET_END
    )
    merged_data = _add_extra_details(merged_data)
    merged_data.to_csv("merged_data.csv", index=False)
