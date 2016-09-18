#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os

# Pynder Imports
import pynder
import urllib
import requests

from wtforms import Form, BooleanField, StringField, PasswordField, validators
#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
#db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Login required decorator.

def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/')
def home():
    return render_template('pages/home.html')

# Match button
@app.route('/users')
def users():
    # get the token from here
    # https://www.facebook.com/dialog/oauth?client_id=464891386855067&redirect_uri=fbconnect%3A%2F%2Fsuccess&scope=basic_info%2Cemail%2Cpublic_profile%2Cuser_about_me%2Cuser_activities%2Cuser_birthday%2Cuser_education_history%2Cuser_friends%2Cuser_interests%2Cuser_likes%2Cuser_location%2Cuser_photos%2Cuser_relationship_details&response_type=token&__mref=message_bubble
    token = "EAAGm0PX4ZCpsBAM2uyxg6P4fhnEnr2zvwnp7uZAQQ7qeoGDcjG8xZBz2I6Yj5vTIpSZA13tXQlxCJJlcoezZAWvEABvZBSUSLzDcLt0jyRjMehmtqCHEr11pB58CJXnTvApHi6iZAU7P7J5GWm9Ki2KDyZAk6g80eeKXRhk8xGNZBlwZDZD"
    id = "591469779"
    session = pynder.Session(id, token)
    print("Session created")
    users = session.nearby_users()
    photos = []
    i=0
    for user in users:
        i+=1
        urllib.urlretrieve(user.get_photos(width='172')[0], 'static/temp'+str(i)+'.jpg')
    target = os.path.join(APP_ROOT, 'static')
    return render_template('pages/users.html')

@app.route('/login')
def login():
    import swipe
    print('done')
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)


@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
