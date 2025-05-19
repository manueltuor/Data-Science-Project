Valuing Players Actions in Football
================================================

This data science project aims to value all football player actions on the pitch. This provides a better view over how different actions contribute to goals, since most models only consider single actions or shots. This project builds up on the paper [Actions Speak Louder than Goals](https://dl.acm.org/doi/pdf/10.1145/3292500.3330758) by Decroos et al. 2019 and the corresponding [Github repository](https://github.com/ML-KULeuven/socceraction). Some ideas were also inspired by Statsbombs [On Ball Value](https://statsbomb.com/news/introducing-on-ball-value-obv/) metric.

General Idea
--------------------

Generally we consider a good action in football as an action that either increases scoring probability or decreases conceding probability or does both. So we can define the action value of an action with the following formula:

**VAEP value(a<sub>i</sub>)** = ΔP<sub>scores</sub>(a<sub>i</sub>) − ΔP<sub>concedes</sub>(a<sub>i</sub>)  
**ΔP<sub>scores</sub>(a<sub>i</sub>)** = P<sub>scores</sub>(S<sub>i</sub>) − P<sub>scores</sub>(S<sub>i−1</sub>)  
**ΔP<sub>concedes</sub>(a<sub>i</sub>)** = P<sub>concedes</sub>(S<sub>i</sub>) − P<sub>concedes</sub>(S<sub>i−1</sub>)

Data
--------------------

We use the **statsbomb** data from the `mplsoccer` python library, this data contains 200 games from the UEFA Euro 2020, FIFA World Cup 2022, 1. Bundesliga 23/24 and UEFA Euro 2024. For these compoetitions the data contains all game states at the time of an action. Each game state contains informations about the action, the acting player and the positions of the other players in the camera frame.

Features
--------------------

We use a variety of features to estimate the probabilities in the model:
- **Distance to Goal:** the distance of the ball to the opponents / own goal
- **Angle to Goal:** the angle of the ball to the opponents / own goal
- **Time Elapsed:** the time elapsed in the game since the start of the game
- **Duration:** the duration of the action
- **Score Difference:** the score difference at the time of the action
- **Body Part:** the body part with which the action was performed
- **Action Type:** the type of the action
- **Closest Defender Distance:** the distance of the ball to the closest defending player
- **Opponents in Front:** the number of opponents in front of the ball with respect to the x coordinate
- **Goalkeeper Distance to Ball:** the distance of the opponents / own goalkeeper to the ball
- **Goalkeeper Distance to Goal:** the distance of the opponents / own goalkeeper to the goal
- **Goalkeeper in Shooting Triangle:** whether or not the opponents / own goalkeepeer is in the triangle between the ball and posts

Labels
--------------------

For probability estimation we label the 10 actions previous to a goal. Actions from the team scoring the goal will receive a 1 for the scoring label and actions from the conceding team will receive a 1 for the conceding label. All other actions will be labelled with a 0.

Model
--------------------

We use an XGBoost classifier to estimate the scoring and conceding probabilities given the features. The model outputs the following:

-   `predicted_goal_prob`: Probability of scoring from the event.

-   `predicted_concede_prob`: Probability the team concedes after the event.

-   `action_value`: Estimated contribution of the action to the overall chance of scoring.

Project Structure and Setup
================================================

Project Structure
--------------------

The files in the project are required to be run in the following order:

-   `1_Loading_Data.ipynb`\
    Loads the data, constructs the features, assigns the labels, computes player minutes and stores everything as `data_cleaned.pkl` and `player_minutes.pkl`.

-   `2_Training_Model.ipynb`\
    Trains the model, estimates the probabilities, computes action values and stores the data in `data_cleaned_trained.pkl`.

-   `3_Result_Visualization.ipynb`\
    Visualizes the model results and shows various potential applications with examples.

Applications
-----------

### 1\. **Player Ratings**

The model can rate all players in the dataset by their action value per 90 minutes.

<img width="295" alt="image" src="https://github.com/user-attachments/assets/6a70b6e3-badc-4356-b10e-aff1456e7c19" />

![image](https://github.com/user-attachments/assets/131cabbe-66ad-4bf2-9bcf-2fe580ff5c21)

### 2\. **Event Snapshot Plotting**

Plots a single event on a pitch, including visible area and freeze-frame positions of players.

![image](https://github.com/user-attachments/assets/b3d1c34e-fd99-42c4-8d78-d7d8b0300263)


`plot_event_with_360(event_row)`

### 3\. **Action Chain Visualization**

Generates a table and pitch plot for a sequence of actions (passes, carries, shots, etc.) leading up to a key moment.

![image](https://github.com/user-attachments/assets/2c40fbb9-4efc-40b2-b102-d1ed3ce9405b)

`plot_action_chain_by_id(df_model, SCENARIO)`

### 4\. **Action Chain Animation**

Creates an animated pitch plot of action sequences with:

-   Player locations

-   Arrows for passes/dribbles

-   Dynamic predicted goal/concede probabilities

https://github.com/user-attachments/assets/8a7ead33-ba92-45cd-ae5e-028421edd40a




`animate_action_chain(df_model, SCENARIO, save_path='action_chain_animation.mp4')`

### 5\. **Team (or Player) Goal Probability Heatmap**

Creates a pitch heatmap showing areas from which actions resulted in high predicted probabilities of scoring.

![image](https://github.com/user-attachments/assets/5d4cc1e4-3095-47cb-9980-37e4c7b56e59)

`plot_team_goal_prob_heatmap(team_id, player_id=None)`

Requirements
---------------

-   `pandas`

-   `matplotlib`

-   `mplsoccer`

-   `numpy`

-   `cmasher`

-   `sklearn`

-   `xgboost`

-   `collections`


Install dependencies with:


`pip install pandas matplotlib mplsoccer numpy cmasher sklearn xgboost collections`
