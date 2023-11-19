from flask import Blueprint, render_template
                             #Library for rendering a template

#This file is the blue print of our application->Urls are define here
#Stores the standard routes for our website->where users can go to

views = Blueprint('views',__name__)   #defining views blueprint

#Defining a route or a url
@views.route('/')        #defining the homepage('/')
def home():              #Function will run whenevever user goes to the main page

     return render_template("home.html")     #returning the html file for rendering