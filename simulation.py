import sys
import pandas as pd
import numpy as np
from markov_chain import MarkovChain

DATA_FILE_NAME = "Probability Project Data.xlsx"

# We need these to decide who wins
ut_score = 0
opp_score = 0


def chain_callback(chain, current_state, num_steps, dec_val):
    global ut_score
    global opp_score
    if current_state == 8:
        # UT scored a field goal
        ut_score += 2
    elif current_state == 9:
        # UT scored a three-pointer
        ut_score += 3
    elif current_state == 9:
        # Repeat for opposing team
        opp_score += 2
    elif current_state == 10:
        opp_score += 3

    # Ensure we can't get ties
    if num_steps <= 0 and ut_score == opp_score:
        return dec_val
    return -1


def run_chain(chain, simulation_length, simulation_step, num_of_games):
    global ut_score
    global opp_score
    ut_wins = 0
    opp_wins = 0

    for i in range(num_of_games):
        ut_score = 0
        opp_score = 0
        state = chain.run(
            simulation_length, np.random.randint(0, 1), simulation_step, chain_callback
        )
        assert state != -1, "Chain encountered an internal error"

        if ut_score > opp_score:
            ut_wins += 1
        elif opp_score > ut_score:
            opp_wins += 1
        # There is no else because callback breaks ties

    return ut_wins, opp_wins


def main(num_iterations):
    dataframe = pd.read_excel(DATA_FILE_NAME)

    # Due to how the excel is structured, in order to calculate things like mean we need to sum every other odd or even row and divide by len(col) / 2
    field_goals_attempted_column = dataframe["Field Goals Attempted"]
    three_pointers_attempted_column = dataframe["3 Pointers Attempted"]
    offensive_rebounds_column = dataframe["Offensive Rebounds"]
    defensive_rebounds_column = dataframe["Defensive Rebounds"]
    average_rebounds_column = dataframe["Average Rebounds per Game"]
    turnovers_column = dataframe["Turnovers"]
    # Number of entries shall stay mostly the same, so we store it
    num_of_individual_entries = len(field_goals_attempted_column) / 2
    number_of_turnovers_ut = (
        sum(turnovers_column[i] for i in range(0, len(turnovers_column), 2))
        / num_of_individual_entries
    )
    number_of_turnovers_opps = (
        sum(turnovers_column[i] for i in range(1, len(turnovers_column), 2))
        / num_of_individual_entries
    )
    number_of_offensive_rebounds_ut = (
        sum(
            offensive_rebounds_column[i]
            for i in range(0, len(offensive_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_offensive_rebounds_opps = (
        sum(
            offensive_rebounds_column[i]
            for i in range(1, len(offensive_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_defensive_rebounds_ut = (
        sum(
            defensive_rebounds_column[i]
            for i in range(0, len(defensive_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_defensive_rebounds_opps = (
        sum(
            defensive_rebounds_column[i]
            for i in range(1, len(defensive_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_average_rebounds_ut = (
        sum(
            average_rebounds_column[i]
            for i in range(0, len(average_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_average_rebounds_opps = (
        sum(
            average_rebounds_column[i]
            for i in range(1, len(average_rebounds_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_field_goal_attempts_ut = (
        sum(
            field_goals_attempted_column[i]
            for i in range(0, len(field_goals_attempted_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_three_ptr_attempts_ut = sum(
        three_pointers_attempted_column[i]
        for i in range(0, len(three_pointers_attempted_column), 2)
    )
    total_shots_ut = number_of_field_goal_attempts_ut + number_of_three_ptr_attempts_ut
    number_of_field_goal_attempts_opps = (
        sum(
            field_goals_attempted_column[i]
            for i in range(1, len(field_goals_attempted_column), 2)
        )
        / num_of_individual_entries
    )
    number_of_three_ptr_attempts_opps = (
        sum(
            three_pointers_attempted_column[i]
            for i in range(1, len(three_pointers_attempted_column), 2)
        )
        / num_of_individual_entries
    )
    total_shots_opps = (
        number_of_field_goal_attempts_opps + number_of_three_ptr_attempts_opps
    )
    likelihood_of_field_attempts_ut = number_of_field_goal_attempts_ut / total_shots_ut
    likelihood_of_three_ptr_attempts_ut = (
        number_of_three_ptr_attempts_ut / total_shots_ut
    )
    likelihood_of_field_attempts_opps = (
        number_of_field_goal_attempts_opps / total_shots_opps
    )
    likelihood_of_three_ptr_attempts_opps = (
        number_of_three_ptr_attempts_opps / total_shots_opps
    )

    field_goal_column = dataframe["Field Goal Percentage"]
    field_goal_column_ut = (
        sum(field_goal_column[i] for i in range(0, len(field_goal_column), 2))
        / num_of_individual_entries
    )
    field_goal_column_opps = (
        sum(field_goal_column[i] for i in range(1, len(field_goal_column), 2))
        / num_of_individual_entries
    )

    three_ptr_column = dataframe["3 Pointer Percentage"]
    three_ptr_column_ut = (
        sum(three_ptr_column[i] for i in range(0, len(three_ptr_column), 2))
        / num_of_individual_entries
    )
    three_ptr_column_opps = (
        sum(three_ptr_column[i] for i in range(1, len(three_ptr_column), 2))
        / num_of_individual_entries
    )
    # The total number of drives is equal to turnovers + attempted 3 pointers + attempted 2 pointers
    total_drives_ut = (
        number_of_turnovers_ut
        + number_of_three_ptr_attempts_ut
        + number_of_field_goal_attempts_ut
    )
    total_drives_opps = (
        number_of_turnovers_opps
        + number_of_three_ptr_attempts_opps
        + number_of_field_goal_attempts_opps
    )

    chain = MarkovChain(14)
    # Each entry in the np.array corresponds to the likelihood of transitioning to that state.
    # This holds true for both teams
    chain.add(
        0,
        "UT possession",
        np.array(
            [
                0,
                number_of_turnovers_ut / total_drives_ut,
                1 - number_of_turnovers_ut / total_drives_ut,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    chain.add(
        1,
        "Enemy possession",
        np.array(
            [
                number_of_turnovers_opps / total_drives_opps,
                0,
                0,
                1 - number_of_turnovers_opps / total_drives_opps,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    chain.add(
        2,
        "UT attempts a shot",
        np.array(
            [
                0,
                0,
                0,
                0,
                likelihood_of_field_attempts_ut,
                likelihood_of_three_ptr_attempts_ut,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    chain.add(
        3,
        "Enemy attempts a shot",
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                likelihood_of_field_attempts_opps,
                likelihood_of_three_ptr_attempts_opps,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    chain.add(
        4,
        "UT goes for a field goal",
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                field_goal_column_ut,
                0,
                0,
                0,
                1 - field_goal_column_ut,
                0,
            ]
        ),
    )

    chain.add(
        5,
        "UT goes for a three pointer",
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                three_ptr_column_ut,
                0,
                0,
                1 - three_ptr_column_ut,
                0,
            ]
        ),
    )

    chain.add(
        6,
        "Enemy goes for a field goal",
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                field_goal_column_opps,
                0,
                0,
                1 - field_goal_column_opps,
            ]
        ),
    )

    chain.add(
        7,
        "Enemy goes for a three pointer",
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                three_ptr_column_opps,
                0,
                1 - three_ptr_column_opps,
            ]
        ),
    )

    # These were not merged because of different scoring possibilities
    # They will, however, have identical probabilities
    chain.add(
        8, "UT makes field goal", np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )

    chain.add(
        9,
        "UT makes three pointer",
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    # These are a repeat of the last 2, except we switch who gets the ball
    chain.add(
        10,
        "Enemy makes field goal",
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    chain.add(
        11,
        "Enemy makes three pointer",
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    ut_average_offensive_rebounds = number_of_average_rebounds_ut * (
        number_of_offensive_rebounds_ut
        / (number_of_offensive_rebounds_ut + number_of_defensive_rebounds_ut)
    )
    opps_average_defensive_rebounds = number_of_average_rebounds_opps * (
        number_of_defensive_rebounds_opps
        / (number_of_offensive_rebounds_opps + number_of_defensive_rebounds_opps)
    )

    chain.add(
        12,
        "UT fails shot",
        np.array(
            [
                ut_average_offensive_rebounds
                / (ut_average_offensive_rebounds + opps_average_defensive_rebounds),
                opps_average_defensive_rebounds
                / (ut_average_offensive_rebounds + opps_average_defensive_rebounds),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    ut_average_defensive_rebounds = number_of_average_rebounds_ut * (
        number_of_defensive_rebounds_ut
        / (number_of_offensive_rebounds_ut + number_of_defensive_rebounds_ut)
    )
    opps_average_offensive_rebounds = number_of_average_rebounds_opps * (
        number_of_offensive_rebounds_opps
        / (number_of_offensive_rebounds_opps + number_of_defensive_rebounds_opps)
    )

    chain.add(
        13,
        "Enemy fails shot",
        np.array(
            [
                ut_average_defensive_rebounds
                / (ut_average_defensive_rebounds + opps_average_offensive_rebounds),
                opps_average_offensive_rebounds
                / (ut_average_defensive_rebounds + opps_average_offensive_rebounds),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    # If you add more states, you will need to expand the matrix by appending zeros
    # If you switch state order, you will need to switch the nonzero probabilities

    games_won_column = dataframe["Games Won"]
    games_lost_column = dataframe["Games Lost"]
    total_games = int(
        np.ceil(
            (
                sum(games_won_column[i] for i in range(0, len(games_won_column), 2))
                + sum(games_lost_column[i] for i in range(0, len(games_lost_column), 2))
            )
            / num_of_individual_entries
        )
    )

    # Careful, we're missing some data for minutes
    # Chop the last 10 not-a-number entries
    minute_column = dataframe["Minutes"][:-10]
    total_minutes = sum(minute_column[i] for i in range(0, len(minute_column), 2)) / (
        len(minute_column) / 2
    )
    # total_minutes is the number of minutes spent playing by all five players, so we divide it by 5
    simulation_length = total_minutes / total_games / 5
    simulation_step = total_minutes / (total_drives_ut + total_drives_opps)

    ut_wins = 0
    ut_losses = 0

    # We run the chain for a sufficiently large number of iterations to get an accurate result
    for i in range(num_iterations):
        wins, losses = run_chain(chain, simulation_length, simulation_step, total_games)
        ut_wins += wins
        ut_losses += losses

    ut_wins = int(np.ceil(ut_wins / num_iterations))
    ut_losses = int(np.ceil(ut_losses / num_iterations))
    print(
        f"After simulating a season {num_iterations} times, UT ended up with {ut_wins} wins and {ut_losses} losses"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <number of chain iterations>")
        exit(-1)

    try:
        num_iterations = int(sys.argv[1])
    except ValueError:
        print("Passed in value must be an integer")
        exit(-1)
    main(num_iterations)
