Action Value Visualization for Football Events
================================================

This project visualizes model predictions of football event values and goal probabilities using real match data. It includes static and animated visualizations that help interpret how sequences of actions contribute to a team's chances of scoring.

Project Structure
--------------------

-   `data/data_cleaned_trained.pkl`\
    A preprocessed dataset with model predictions, including action values and predicted goal probabilities.

-   `2_Training_Model.ipynb`\
    (Required to be run beforehand) Trains the model and stores the data in `data_cleaned_trained.pkl`.

Features
-----------

### 1\. **Event Snapshot Plotting**

Plots a single event on a pitch, including visible area and freeze-frame positions of players.

`plot_event_with_360(event_row)`

### 2\. **Action Chain Visualization**

Generates a table and pitch plot for a sequence of actions (passes, carries, shots, etc.) leading up to a key moment.

`plot_action_chain_by_id(df_model, SCENARIO)`

### 3\. **Action Chain Animation**

Creates an animated pitch plot of action sequences with:

-   Player locations

-   Arrows for passes/dribbles

-   Dynamic predicted goal/concede probabilities

`animate_action_chain(df_model, SCENARIO, save_path='action_chain_animation.mp4')`

### 4\. **Team (or Player) Goal Probability Heatmap**

Creates a pitch heatmap showing areas from which actions resulted in high predicted probabilities of scoring.


`plot_team_goal_prob_heatmap(team_id, player_id=None)`

Model Output
---------------

The model enriches each event in the dataset with:

-   `predicted_goal_prob`: Probability of scoring from the event.

-   `predicted_concede_prob`: Probability the team concedes after the event.

-   `action_value`: Estimated contribution of the action to the overall chance of scoring.

Requirements
---------------

-   `pandas`

-   `matplotlib`

-   `mplsoccer`

-   `numpy`

-   `cmasher`

Install dependencies with:


`pip install pandas matplotlib mplsoccer numpy cmasher`

Notes
--------

-   **Run the training notebook (`2_Training_Model.ipynb`) first** to generate the required `.pkl` file.

-   Use valid event IDs to visualize specific game sequences. For example:

    -   `"57c5911b-9d18-4c11-8975-c73e81ed0940"` for *Cameroon vs Brazil*

    -   `"17949ad2-d653-49cb-95e4-2efe585438dc"` for *Japan Comeback*
