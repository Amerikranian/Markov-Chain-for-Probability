import pandas as pd
import numpy as np
from markov_chain import MarkovChain

DATA_FILE_NAME = "Probability Project Data.xlsx"


def main():
    dataframe = pd.read_excel(DATA_FILE_NAME)

    # Due to how the excel is structured, in order to calculate things like mean we need to sum every other odd or even row and divide by len(col) / 2
    field_goals_attempted_column = dataframe["Field Goals Attempted"]
    three_pointers_attempted_column = dataframe["3 Pointers Attempted"]
    # Number of entries shall stay mostly the same, so we store it
    num_of_individual_entries = len(field_goals_attempted_column) / 2
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

    chain = MarkovChain(14)
    # Each entry in the np.array corresponds to the likelihood of transitioning to that state.
    # In this case entry 2 should be derived from 1-p_(keeps possession)-p_(loses possession)
    # This holds true for both teams
    chain.add(
        0, "UT possession", np.array([-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    chain.add(
        1, "Enemy possession", np.array([-1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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

    chain.add(
        12, "UT fails shot", np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    chain.add(
        13, "Enemy fails shot", np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )

    # If you add more states, you will need to expand the matrix by appending zeros
    # If you switch state order, you will need to switch the nonzero probabilities

    # Todo: Add code to run / print stuff about Markov chain when the probabilities are fixed


if __name__ == "__main__":
    main()
