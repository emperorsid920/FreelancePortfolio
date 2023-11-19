from flask import Blueprint, render_template, request, flash



auth = Blueprint('auth',__name__)   #defining auth blueprint

#defining the the routes
@auth.route('/login', methods =['GET', 'POST'])              #login page def
def login():
    data = request.form         #retreive the data sent to the forms
    return render_template("login.html")

@auth.route('/logout')            #Logiut page def
def logout():
    return "<p>Logout</p>"

@auth.route('/sign-up',  methods =['GET', 'POST'])           #Sign Up page def
def sign_up():

    #differentiate between post and get
    if request.method == 'POST':
        email = request.form.get('email')
        firstName = request.form.get('firstName')
        password1 = request.form.get('password1')  # Corrected field name
        password2 = request.form.get('password2')  # Corrected field name

        #Checking Sign up credentials
        if len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(firstName) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            flash('Account Created!!', category='success')

    return render_template("sign_up.html")

