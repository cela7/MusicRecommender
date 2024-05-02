--- Music Recommender ---

! This repo does not contain the dataset used for this project and must be sourced elsewhere !

To setup, do the following in order:
Download the Spotify Million Playlist Dataset
Enter your Spotify client id and client secret in the credentials.txt file
python prepareData.py <path to where the Spotify Million Playlist Dataset is located>
python cf.py
python cnn.py (This step is very long and saves a checkpoint every 3 iterations. The
entire 50 iterations can be cut short midway through and progress will be saved)
python xgb.py

Afterwards, to launch the recommender, run the following:
python main.py
or use:
python main.py verbose

This will ask for a Spotify username, which will open a webpage to allow access to
your account. Follow the instructions of the program to complete this step.
Once this is done, the program will ask for the id to a playlist. This can be
found by right clicking any Spotify playlist, hovering over "Share", and clicking
"Copy link to playlist". The id can then be found within the link after
"https://open.spotify.com/playlist/" and before the "?" symbol.

To run metrics, simply run the following:
python main.py metrics