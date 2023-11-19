from flask import Flask
from flask_sqlalchemy import SQLAlchemy

#Defining a new database
db = SQLAlchemy()         #initializing the database
DB_NAME = "database.db"   #Naming the database

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'FHIUHFI872EJ'

    #Configuring the location of the database.
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    #initializing the database to the flask app
    db.init_app(app)



    from .views import views      #importing blueprint from views
    from .auth import auth        #importing blueprint from auth

    # registering the blueprints to the flask application
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app
