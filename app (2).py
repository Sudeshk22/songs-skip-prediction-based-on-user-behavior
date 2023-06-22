from flask import Flask, render_template, request
import pickle

import numpy as np

lModel = pickle.load(open('random_forest_Model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/submit', methods=["POST"])
def submit():
    duration = float(request.form["duration"])
    acousticness = float(request.form["acousticness"])
    speechiness = float(request.form["speechiness"])
    acoustic_vector_0 = float(request.form["acoustic_vector_0"])
    acoustic_vector_2 = float(request.form["acoustic_vector_2"])
    acoustic_vector_4 = float(request.form["acoustic_vector_4"])
    acoustic_vector_6 = float(request.form["acoustic_vector_6"])
    session_position = float(request.form["session_position"])
    session_length = float(request.form["session_length"])
    context_switch = float(request.form["context_switch"])
    no_pause_before_play = float(request.form["no_pause_before_play"])
    short_pause_before_play = float(request.form["short_pause_before_play"])
    long_pause_before_play = float(request.form["long_pause_before_play"])
    hist_user_behavior_is_shuffle = float(
        request.form["hist_user_behavior_is_shuffle"])
    premium = float(request.form["premium"])
    context_type = float(request.form["context_type"])
    hist_user_behavior_reason_start = float(
        request.form["hist_user_behavior_reason_start"])
    hist_user_behavior_reason_end = float(
        request.form["hist_user_behavior_reason_end"])
    acoustic_vector_pca1 = float(request.form["acoustic_vector_pca1"])
    acoustic_vector_pca3 = float(request.form["acoustic_vector_pca3"])

    predictedVal = lModel.predict(np.array([duration, acousticness, speechiness, acoustic_vector_0, acoustic_vector_2, acoustic_vector_4, acoustic_vector_6, session_position, session_length, context_switch, no_pause_before_play,
                                  short_pause_before_play, long_pause_before_play, context_type, premium, hist_user_behavior_is_shuffle, hist_user_behavior_reason_start, hist_user_behavior_reason_end, acoustic_vector_pca1, acoustic_vector_pca3]).reshape(1, 20))
    if predictedVal[0] == 1:
        predictedVal = 'skipped track'
    else:
        predictedVal = 'Track Not Skipped'

    return render_template("index.html", prediction=predictedVal)
if __name__ == '__main__':
    app.run()