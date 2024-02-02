const express = require("express");
const cors = require("cors");
const db = require("./app/models");
const cryptos = require("./app/controllers/crypto.controller");
const bcrypt = require("bcryptjs")
const jwt = require("jsonwebtoken");
const tokenSecret = "my-token-secret";
const users = require("./app/controllers/users.controller");
const User = db.users;
const app = express();
const session = require('express-session');
const passport = require('passport');
var corsOptions = {
    origin: "http://localhost:8081"
};
var userProfile;
const GoogleStrategy = require('passport-google-oauth').OAuth2Strategy;
const GOOGLE_CLIENT_ID = '639415844417-313sq6lcq9iempnofor2l03bgq098u8f.apps.googleusercontent.com';
const GOOGLE_CLIENT_SECRET = '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';

app.set('view engine', 'ejs');
app.use(session({
    resave: false,
    saveUninitialized: true,
    secret: 'SECRET'
}));

app.use(passport.initialize());
app.use(passport.session());
app.get('/error', (req, res) => res.send("error logging in"));

passport.serializeUser(function (user, cb) {
    cb(null, user);
});

passport.deserializeUser(function (obj, cb) {
    cb(null, obj);
});

passport.use(new GoogleStrategy({
    clientID: GOOGLE_CLIENT_ID,
    clientSecret: GOOGLE_CLIENT_SECRET,
    callbackURL: "http://localhost:4000/auth/google/callback"
},
    function (accessToken, refreshToken, profile, done) {
        userProfile = profile;
        return done(null, userProfile);
    }
));

app.get('/auth/google',
    passport.authenticate('google', { scope: ['profile', 'email'] }));

app.get('/auth/google/callback',
    passport.authenticate('google', { failureRedirect: '/error' }),
      function (req, res) {
        User.findOne({ where: { email: userProfile._json.email } }).then(user => {
            let token = ""
            if (!user)
                this.createOauth(req, res);
            else
                bcrypt.compare(userProfile.id, user.password, (error, match) => {
                    if (error) {
                        res.status(500).json(error)
                    }
                    else if (match) {
                        token = jwt.sign({ data: user }, tokenSecret, { expiresIn: ('24h') })
                        res.status(200).json({ token: token })
                    }
                    else {
                        res.status(403).json({ error: 'passwords do not match' })
                    }
                })
        }).catch(error => {
            res.status(500).json(error)
        })
    });

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

db.sequelize.sync()
    .then(() => {
        console.log("Synced db.");
    })
    .catch((err) => {
        console.log("Failed to sync db: " + err.message);
    });

app.get("/", (req, res) => {
    res.json({ message: "Welcome to the count_of_money application." });
});

require("./app/routes/crypto.routes")(app);
require("./app/routes/users.routes")(app);
require("./app/routes/user_crypto.routes")(app);
require("./app/routes/crypto_value.routes")(app);
require("./app/routes/rss.routes")(app);
require("./app/routes/user_rss.routes")(app);
require("./app/routes/admin_settings.routes")(app);

setInterval(cryptos.Find5First, 300000)
setInterval(cryptos.UpdateShowCrypto, 60000)

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}.`);
});