const loggingBack = document.getElementById("loggingBack");
const Lemail = document.getElementById("Lemail");
const Lpassword = document.getElementById("Lpassword");
const Lsubmit = document.getElementById("Lsubmit");
const Lregister = document.getElementById("Lregister");
let username = document.getElementById("Luser");;

Lregister.onclick = function() {
    registerBack.style.display = "block";
}


const registerBack = document.getElementById("registerBack");
const Rx = document.getElementById("Rx");
const Remail = document.getElementById("Remail");
const Rpassword = document.getElementById("Rpassword");
const Rsubmit = document.getElementById("Rsubmit");

Rx.onclick = function() {
    registerBack.style.display = "none";
}

function hide (event) {
    if (event.target == registerBack)
        registerBack.style.display = "none";
}
window.addEventListener('click', hide, false);


const OWeather = document.getElementById("OWeather");
const pOWeather = document.getElementById("+OWeather");
const OYoutube = document.getElementById("OYoutube");
const pOYoutube = document.getElementById("+OYoutube");
const OConversion = document.getElementById("OConversion");
const pOConversion = document.getElementById("+OConversion");
const OGameFinder = document.getElementById("OGameFinder");
const pOGameFinder = document.getElementById("+OGameFinder");
const OFriendList = document.getElementById("OFriendList");
const pOFriendList = document.getElementById("+OFriendList");
const OHistory = document.getElementById("OHistory");
const pOHistory = document.getElementById("+OHistory");
const board = document.getElementById("board");


pOFriendList.onclick = function() {
    let Bblock = createWidget("FriendList");
    let Ftop = document.createElement("div");
    let Fbot = document.createElement("div");
    Ftop.setAttribute("class", "Ftop");
    Fbot.setAttribute("class", "Fbot");

    let FriendI = document.createElement("input");
    let FriendB = document.createElement("button");
    FriendI.setAttribute("type", "text");
    FriendI.setAttribute("name", "Hname");
    FriendI.setAttribute("placeholder", "Enter player's URL vanity");
    FriendB.setAttribute("type", "button");
    FriendB.innerHTML = "Find";

    Bblock.appendChild(Ftop);
    Bblock.appendChild(Fbot);
    Fbot.appendChild(FriendI);
    Fbot.appendChild(FriendB);

    FriendB.onclick = function() {
        let request = new XMLHttpRequest();
        let clean = Bblock.getElementsByClassName("FtopContent")

        if (clean.length > 0)
            clean[0].parentNode.removeChild(clean[0]);
        let FtopContent = document.createElement("div");
        FtopContent.setAttribute("class", "FtopContent");
        Ftop.appendChild(FtopContent);
        
        request.open("GET", "http://localhost:8080/steam/friendlist?name=" + FriendI.value, true);
        request.onload = function () {
            let data = JSON.parse(this.response);
    
            if (request.status == 200) {

                for (let i = 0; i < data.Names.length ; i++) {
                    let Fp = document.createElement("p");
                    Fp.setAttribute("id", "Fp");
                    Fp.innerHTML = data.Names[i];
                    console.log(data.Names[i]);
                    FtopContent.appendChild(Fp);
                }
            } else
                console.log('error ${request.status} ${request.statusText}');
        }
        request.send();
    }
}




pOHistory.onclick = function() {
    let Bblock = createWidget("History");
    let Htop = document.createElement("div");
    let Hbot = document.createElement("div");
    Htop.setAttribute("class", "Htop");
    Hbot.setAttribute("class", "Hbot");

    let HistoryI = document.createElement("input");
    let HistoryB = document.createElement("button");
    HistoryI.setAttribute("type", "text");
    HistoryI.setAttribute("name", "Hname");
    HistoryI.setAttribute("placeholder", "Enter player's URL vanity");
    HistoryB.setAttribute("type", "button");
    HistoryB.innerHTML = "Find";

    Bblock.appendChild(Htop);
    Bblock.appendChild(Hbot);
    Hbot.appendChild(HistoryI);
    Hbot.appendChild(HistoryB);

    HistoryB.onclick = function() {
        let request = new XMLHttpRequest();
        let clean = Bblock.getElementsByClassName("HtopContent")

        if (clean.length >0)
            clean[0].parentNode.removeChild(clean[0]);
        let HtopContent = document.createElement("div");
        HtopContent.setAttribute("class", "HtopContent");
        Htop.appendChild(HtopContent);
        
        request.open("GET", "http://localhost:8080/steam/playedGames?name=" + HistoryI.value, true);
        request.onload = function () {
            let data = JSON.parse(this.response);
    
            if (request.status == 200) {

                for (let i = 0; i < data.response.games.length; i++) {
                    let Hcontainer = document.createElement("div");
                    let Himg = document.createElement("img");
                    let Hp = document.createElement("p");
                    let Hp2 = document.createElement("p");
                    Hp.setAttribute("id", "Hp");
                    Hp2.setAttribute("id", "Hp2")
                    Hcontainer.setAttribute("id", "Hcontainer");
                    Himg.setAttribute("id", "Himg");
                    Hp.innerHTML = data.response.games[i].name;
                    Hp2.innerHTML = "playtime 2weeks: " + (data.response.games[i].playtime_2weeks / 60).toFixed(1) + "h";
                    Himg.src = "https://cdn.cloudflare.steamstatic.com/steamcommunity/public/images/apps/" + data.response.games[i].appid + "/" + data.response.games[i].img_logo_url  + ".jpg";
                    HtopContent.appendChild(Hcontainer);
                    Hcontainer.appendChild(Himg);
                    Hcontainer.appendChild(Hp);
                    Hcontainer.appendChild(Hp2);
                }
            } else
                console.log('error ${request.status} ${request.statusText}');
        }
        request.send();
    }
}

pOWeather.onclick = function() {
    
    let Bblock = createWidget("Weather");
    let Wtop = document.createElement("div");
    let Wmid = document.createElement("div");
    let Wmid1 = document.createElement("div");
    let Wmid2 = document.createElement("div");
    let Wmid3 = document.createElement("div");
    let Wmid4 = document.createElement("div");
    let Wbot = document.createElement("div");
    Wtop.setAttribute("class", "Wtop")
    Wmid.setAttribute("class", "Wmid");
    Wbot.setAttribute("class", "Wbot");

    let cityI = document.createElement("input");
    let cityB = document.createElement("button");
    cityI.setAttribute("type", "text");
    cityI.setAttribute("name", "Wcity");
    cityI.setAttribute("placeholder", "Find a city");
    cityB.setAttribute("type", "button");
    cityB.innerHTML = "Find";

    let WLocation = document.createElement("p");
    let Wsunrise = document.createElement("p");
    let Whumidity = document.createElement("p");
    let Wcondition = document.createElement("p");
    let Wday1 = document.createElement("h2");
    let Wcondition1 = document.createElement("p");
    let Wmax1 = document.createElement("p");
    let Wday2 = document.createElement("h2");
    let Wcondition2 = document.createElement("p");
    let Wmax2 = document.createElement("p");
    let Wday3 = document.createElement("h2");
    let Wcondition3 = document.createElement("p");
    let Wmax3 = document.createElement("p");
    let Wday4 = document.createElement("h2");
    let Wcondition4 = document.createElement("p");
    let Wmax4 = document.createElement("p");
    WLocation.style.fontSize = "x-large";
    Wcondition.style.fontSize = "xx-large";
    Wcondition.style.marginTop = "55px";
    Wcondition.style.marginLeft = "0px";

    Bblock.appendChild(Wtop);
    Wtop.appendChild(Wsunrise);
    Wtop.appendChild(Whumidity);
    Wtop.appendChild(Wcondition);
    Wtop.appendChild(WLocation);
    Bblock.appendChild(Wmid);
    Wmid.appendChild(Wmid1);
    Wmid1.appendChild(Wday1);
    Wmid1.appendChild(Wcondition1);
    Wmid1.appendChild(Wmax1);
    Wmid.appendChild(Wmid2);
    Wmid2.appendChild(Wday2);
    Wmid2.appendChild(Wcondition2);
    Wmid2.appendChild(Wmax2);
    Wmid.appendChild(Wmid3);
    Wmid3.appendChild(Wday3);
    Wmid3.appendChild(Wcondition3);
    Wmid3.appendChild(Wmax3);
    Wmid.appendChild(Wmid4);
    Wmid4.appendChild(Wday4);
    Wmid4.appendChild(Wcondition4);
    Wmid4.appendChild(Wmax4);
    Bblock.appendChild(Wbot);
    Wbot.appendChild(cityI);
    Wbot.appendChild(cityB);

    cityB.onclick = function() {
        let request = new XMLHttpRequest();
        
        request.open("GET", "http://localhost:8080/weather?city=" + cityI.value.replaceAll(' ', ''), true);
        request.onload = function () {
            let data = JSON.parse(this.response);
    
            if (request.status == 200) {
                WLocation.innerHTML = data.location.city + ", " + data.location.region + ", " + data.location.country;
                Wsunrise.innerHTML = "Sunrise: " + data.current_observation.astronomy.sunrise + "<br>Sunset: " + data.current_observation.astronomy.sunset;
                Whumidity.innerHTML = "humidity: " + data.current_observation.atmosphere.humidity + "%";
                Wcondition.innerHTML = data.current_observation.condition.text + ", " + ((data.current_observation.condition.temperature - 32) * (5/9)).toFixed(1) + "°C";
                Wday1.innerHTML = data.forecasts[0].day;
                Wcondition1.innerHTML = data.forecasts[1].text;
                Wmax1.innerHTML = ((data.forecasts[0].low - 32) * (5/9)).toFixed(0) + " - " + ((data.forecasts[0].high - 32) * (5/9)).toFixed(0) + "°C";
                Wday2.innerHTML = data.forecasts[1].day;
                Wcondition2.innerHTML = data.forecasts[1].text;
                Wmax2.innerHTML = ((data.forecasts[1].low - 32) * (5/9)).toFixed(0) + " - " + ((data.forecasts[1].high - 32) * (5/9)).toFixed(0) + "°C";
                Wday3.innerHTML = data.forecasts[2].day;
                Wcondition3.innerHTML = data.forecasts[2].text;
                Wmax3.innerHTML = ((data.forecasts[2].low - 32) * (5/9)).toFixed(0) + " - " + ((data.forecasts[2].high - 32) * (5/9)).toFixed(0) + "°C";
                Wday4.innerHTML = data.forecasts[3].day;
                Wcondition4.innerHTML = data.forecasts[3].text;
                Wmax4.innerHTML = ((data.forecasts[3].low - 32) * (5/9)).toFixed(0) + " - " + ((data.forecasts[3].high - 32) * (5/9)).toFixed(0) + "°C";;
            } else
                console.log('error ${request.status} ${request.statusText}');
        }
        request.send();
        }

}

pOYoutube.onclick = function() {
    let Bblock = createWidget("Youtube");
    let Ytop = document.createElement("div");
    let Ymid = document.createElement("div");
    let Ymid1 = document.createElement("div");
    let Ymid2 = document.createElement("div");
    let Ymid3 = document.createElement("div");
    let Ybot = document.createElement("div");
    Ytop.setAttribute("class","Ytop");
    Ymid.setAttribute("class","Ymid")
    Ybot.setAttribute("class","Ybot");

    let Yname = document.createElement("h2");
    let Ydescription = document.createElement("p");
    let Ysubs = document.createElement("p");
    let Yviews = document.createElement("p");
    let Ymovies = document.createElement("p");
    let Ysubs0 = document.createElement("h2");
    let Yviews0 = document.createElement("h2");
    let Ymovies0 = document.createElement("h2");
    Ysubs0.innerHTML = "Subscribers";
    Ymovies0.innerHTML = "Video";
    Yviews0.innerHTML = "Views";

    let searchInput = document.createElement("input");
    let searchButton = document.createElement("button");
    searchInput.setAttribute("type", "text");
    searchInput.setAttribute("name", "searchInput");
    searchInput.setAttribute("placeholder", "Find a channel");
    searchButton.setAttribute("type", "button");
    searchButton.innerHTML = "Find";

    searchButton.onclick = function() {
        let request = new XMLHttpRequest();

        request.open("GET", "http://localhost:8080/youtube/id?name=" + searchInput.value.replaceAll(' ', ''), true);
        
        request.onload = function () {
            let data = JSON.parse(this.response);
                if (request.status == 200) {
                    Yname.innerHTML = data.items[0].snippet.localized.title;
                    Ydescription.innerHTML = data.items[0].snippet.localized.description.replaceAll('?', '\'');
                    Ysubs.innerHTML = data.items[0].statistics.subscriberCount.replaceAll(/\B(?=(\d{3})+(?!\d))/g, " ");
                    Yviews.innerHTML = data.items[0].statistics.viewCount.replaceAll(/\B(?=(\d{3})+(?!\d))/g, " ");
                    Ymovies.innerHTML = data.items[0].statistics.videoCount.replaceAll(/\B(?=(\d{3})+(?!\d))/g, " ");
                } else 
                    Yname.innerHTML = "pls Retry";
        }
        request.send();
    }

    Bblock.appendChild(Ytop);
    Ytop.appendChild(Yname);
    Ytop.appendChild(Ydescription);
    Bblock.appendChild(Ymid);
    Ymid.appendChild(Ymid1);
    Ymid.appendChild(Ymid2);
    Ymid.appendChild(Ymid3);
    Ymid1.appendChild(Ysubs0);
    Ymid2.appendChild(Yviews0);
    Ymid3.appendChild(Ymovies0);
    Ymid1.appendChild(Ysubs);
    Ymid2.appendChild(Yviews);
    Ymid3.appendChild(Ymovies);
    Bblock.appendChild(Ybot);
    Ybot.appendChild(searchInput);
    Ybot.appendChild(searchButton);
}

pOGameFinder.onclick = function() {
    let Bblock = createWidget("GameFinder");
    let Gtop = document.createElement("div");
    let Gmid = document.createElement("div");
    let Gmid1 = document.createElement("div");
    let Gmid2 = document.createElement("div");
    let Gmid3 = document.createElement("div");
    let Gbot = document.createElement("div");
    Gtop.setAttribute("class","Gtop");
    Gmid.setAttribute("class","Gmid")
    Gbot.setAttribute("class","Gbot");

    let Gname = document.createElement("h2");
    let Gpublisher = document.createElement("p");
    let Gdeveloper = document.createElement("p");
    let Gdate = document.createElement("p");
    let Gscore = document.createElement("p");
    let Gplayers = document.createElement("p");
    let Gdlc = document.createElement("p");
    let Gscore0 = document.createElement("h2");
    let Gplayers0 = document.createElement("h2");
    let Gdlc0 = document.createElement("h2");
    Gscore0.innerHTML = "MetaCritic";
    Gdlc0.innerHTML = "DLC";
    Gplayers0.innerHTML = "Online";

    let searchInput = document.createElement("input");
    let searchButton = document.createElement("button");
    searchInput.setAttribute("type", "text");
    searchInput.setAttribute("name", "searchInput");
    searchInput.setAttribute("placeholder", "Find a Game");
    searchButton.setAttribute("type", "button");
    searchButton.innerHTML = "Find";

    searchButton.onclick = function() {
        let request = new XMLHttpRequest();

        request.open("GET", "http://localhost:8080/steam/game?name=" + searchInput.value.replaceAll(' ', '%20'), true);
        
        request.onload = function () {
             console.log("reponse = ");
             console.log(this.response);
            let data = JSON.parse(this.response);
            console.log("data = ");
            console.log(data);
                if (request.status == 200) {
                    Gname.innerHTML = data.Game.name
                    Gpublisher.innerHTML = "Publisher: " + data.Game.publishers;
                    Gdeveloper.innerHTML = "Developer: " + data.Game.developers;
                    Gdate.innerHTML = "Release data: " + data.Game.release_date;
                    Gscore.innerHTML = data.Game.metacritic;
                    Gplayers.innerHTML = data.Game.nb.replaceAll(/\B(?=(\d{3})+(?!\d))/g, " ");
                    Gdlc.innerHTML = data.Game.dlc.replaceAll(/\B(?=(\d{3})+(?!\d))/g, " ");
                } else 
                    Gname.innerHTML = "pls Retry";
        }
        request.send();
    }

    Bblock.appendChild(Gtop);
    Gtop.appendChild(Gname);
    Gtop.appendChild(Gpublisher);
    Gtop.appendChild(Gdeveloper);
    Gtop.appendChild(Gdate);
    Bblock.appendChild(Gmid);
    Gmid.appendChild(Gmid1);
    Gmid.appendChild(Gmid2);
    Gmid.appendChild(Gmid3);
    Gmid1.appendChild(Gscore0);
    Gmid2.appendChild(Gplayers0);
    Gmid3.appendChild(Gdlc0);
    Gmid1.appendChild(Gscore);
    Gmid2.appendChild(Gplayers);
    Gmid3.appendChild(Gdlc);
    Bblock.appendChild(Gbot);
    Gbot.appendChild(searchInput);
    Gbot.appendChild(searchButton);
}


pOConversion.onclick = function() {
    let Bblock = createWidget("Conversion");
    let Ctop = document.createElement("div");
    let Cbot = document.createElement("div");
    Ctop.setAttribute("class", "Ctop");
    Cbot.setAttribute("class", "Cbot");
    
    let result = document.createElement("p");
    result.innerHTML = "result";

    let amountInput = document.createElement("input");
    let fromCurrency = document.createElement("select");
    let toCurrency = document.createElement("select");
    let convertButton = document.createElement("button");
    let request = new XMLHttpRequest();
    amountInput.setAttribute("placeholder", "amount");
    fromCurrency.setAttribute("name", "fromCurrency");
    toCurrency.setAttribute("name", "toCurrency");
    convertButton.setAttribute("type", "button");
    convertButton.innerHTML = "convert";
    request.open("GET", "http://localhost:8080/get/currency", true);
    request.onload = function () {
        let data = JSON.parse(this.response);

        for (let i = 0; i < Object.keys(data).length; i++) {
            let option = document.createElement("option");
            let option2 = document.createElement("option");
            option.value = Object.keys(data)[i];
            option.text = Object.keys(data)[i];
            option2.value = Object.keys(data)[i];
            option2.text = Object.keys(data)[i];
            toCurrency.appendChild(option);
            fromCurrency.appendChild(option2);
        }
    }
    request.send();

    convertButton.onclick = function() {
        let request2 = new XMLHttpRequest();

        request2.open("GET", "http://localhost:8080/currency?currency1=" + fromCurrency.value + "&currency2=" + toCurrency.value + "&value=" + amountInput.value.replaceAll(' ', ''), true);

        request2.onload = function () {
            let data = JSON.parse(this.response);
            result.innerHTML = data.toFixed(2);
        }
        request2.send();
    }

    Bblock.appendChild(Ctop);
    Ctop.appendChild(amountInput);
    Ctop.appendChild(fromCurrency);
    Ctop.appendChild(toCurrency);
    Ctop.appendChild(convertButton);
    Bblock.appendChild(Cbot);
    Cbot.appendChild(result);
}

function createWidget(titles) {
    let block = document.createElement("div");
    let Hblock = document.createElement("div");
    let title = document.createElement("h1");
    let Bblock = document.createElement("div");
    let close = document.createElement("button")
    let goBefore = document.createElement("button");
    let goAfter = document.createElement("button");

        block.setAttribute("id", "block" + board.getElementsByClassName('widget').length);
        block.setAttribute("class", "widget")
        Hblock.setAttribute("id", "Hblock" + board.getElementsByClassName('widget').length);
        Hblock.setAttribute("class", "Hblock");
        Bblock.setAttribute("id", "Bblock" + board.getElementsByClassName('widget').length);
        Bblock.setAttribute("class", "Bblock");
        title.innerHTML = titles;
        close.setAttribute("type", "button");
        close.setAttribute("class", "Hclose");
        close.innerHTML = "x";
        goBefore.setAttribute("type", "button");
        goBefore.setAttribute("class", "Bmove");
        goBefore.innerHTML = "<";
        goAfter.setAttribute("type", "button");
        goAfter.setAttribute("class", "Bmove");
        goAfter.innerHTML = ">";
    board.appendChild(block);
    block.appendChild(Hblock);
    Hblock.appendChild(goBefore);
    Hblock.appendChild(goAfter);
    Hblock.appendChild(close);
    Hblock.appendChild(title);
    block.appendChild(Bblock);
   
    goAfter.onclick = function() {
        if(block.nextElementSibling)
            block.parentNode.insertBefore(block, block.nextElementSibling.nextElementSibling);
    }

    goBefore.onclick = function() {
        if(block.previousElementSibling)
            block.parentNode.insertBefore(block, block.previousElementSibling);
    }

    close.onclick = function() {
        board.removeChild(block);
    }
    return Bblock;
}