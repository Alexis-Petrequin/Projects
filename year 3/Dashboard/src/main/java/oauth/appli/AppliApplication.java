package oauth.appli;

import java.util.*;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import org.springframework.context.annotation.Configuration;
import kong.unirest.json.*;

import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;

import java.io.*;

@EnableWebSecurity
@Configuration
@SpringBootApplication
@RestController
public class AppliApplication extends WebSecurityConfigurerAdapter {

	@RequestMapping("/user")
	public Map<String, Object> user(@AuthenticationPrincipal OAuth2User principal) {
		return Collections.singletonMap("name", principal.getAttribute("name"));
	}

	@Override
	protected void configure(HttpSecurity http) throws Exception {
		http.csrf().disable().oauth2Login();
	}

	@GetMapping("/weather")
	public String getWeather(@RequestParam(value = "city", defaultValue = "Strasbourg") String city) {
		try {
			HttpRequest request = HttpRequest.newBuilder()
					.uri(URI.create(
							"https://yahoo-weather5.p.rapidapi.com/weather?location=" + city + "&format=json&u=f"))
					.header("x-rapidapi-host", "yahoo-weather5.p.rapidapi.com")
					.header("x-rapidapi-key", "fb0aca0624msh51b481ce32894cbp18bf4bjsn8c4e9f1912b2")
					.method("GET", HttpRequest.BodyPublishers.noBody())
					.build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());
			System.out.println(response.body());
			return response.body();
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/steam/game")
	public String getSteamGame(@RequestParam(value = "name", defaultValue = "dota 2") String str) {
		String rez = "{\"Game\":";
		try {
			HttpRequest request = HttpRequest.newBuilder()
					.uri(URI.create("https://api.steampowered.com/ISteamApps/GetAppList/v2/"))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());
			JSONObject obj = new JSONObject(response.body());

			final JSONArray apps = obj.getJSONObject("applist").getJSONArray("apps");
			final int n = apps.length();
			for (int i = 0; i < n; ++i) {
				final JSONObject person = apps.getJSONObject(i);
				if (person.getString("name").toLowerCase().equals(str.toLowerCase())) {
					HttpRequest players = HttpRequest.newBuilder().uri(URI.create(
							"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?key=CFB8C146F9C06DC39C650F36C531D6FB&appid="
									+ person.getInt("appid")))
							.method("GET", HttpRequest.BodyPublishers.noBody()).build();
					HttpResponse<String> number = HttpClient.newHttpClient().send(players,
							HttpResponse.BodyHandlers.ofString());

					HttpRequest gameinfo = HttpRequest.newBuilder().uri(URI.create(
							"https://store.steampowered.com/api/appdetails/?appids=" + person.getInt("appid")))
							.method("GET", HttpRequest.BodyPublishers.noBody()).build();
					HttpResponse<String> info = HttpClient.newHttpClient().send(gameinfo,
							HttpResponse.BodyHandlers.ofString());
					rez += "{\"name\":\"" + person.getString("name") + "\",";
					if (new JSONObject(number.body()).getJSONObject("response").getString("result").equals("1")) {
						rez += "\"nb\":\""
								+ new JSONObject(number.body()).getJSONObject("response").getString("player_count")
								+ "\",";
					} else
						rez += "\"nb\":\"0\",";
					int x = 0;
					if (new JSONObject(info.body()).getJSONObject(person.getString("appid")).getJSONObject("data")
							.has("dlc"))
						for (x = 0; x < new JSONObject(info.body()).getJSONObject(person.getString("appid"))
								.getJSONObject("data").getJSONArray("dlc").length(); x++)
							;
					rez += "\"dlc\":\"" + x + "\",";
					if (new JSONObject(info.body()).getJSONObject(person.getString("appid")).getJSONObject("data")
							.has("metacritic")) {
						rez += "\"metacritic\":\""
								+ new JSONObject(info.body()).getJSONObject(person.getString("appid"))
										.getJSONObject("data").getJSONObject("metacritic").getString("score")
								+ "\",";
					} else
						rez += "\"metacritic\":\"n/a\",";
					for (x = 0; x < new JSONObject(info.body()).getJSONObject(person.getString("appid"))
							.getJSONObject("data").getJSONArray("developers").length(); x++)
						rez += "\"developers\":\""
								+ new JSONObject(info.body()).getJSONObject(person.getString("appid"))
										.getJSONObject("data").getJSONArray("developers").getString(x)
								+ "\",";
					for (x = 0; x < new JSONObject(info.body()).getJSONObject(person.getString("appid"))
							.getJSONObject("data").getJSONArray("publishers").length(); x++)
						rez += "\"publishers\":\""
								+ new JSONObject(info.body()).getJSONObject(person.getString("appid"))
										.getJSONObject("data").getJSONArray("publishers").getString(x)
								+ "\",";

					rez += "\"release_date\":\"" + new JSONObject(info.body()).getJSONObject(person.getString("appid"))
							.getJSONObject("data").getJSONObject("release_date").getString("date") + "\"";
					rez += "}}";
					return rez;
				}
			}
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/steam/playedGames")
	public String getSteamPlayedGames(@RequestParam(value = "name", defaultValue = "VDSDU25") String str) {
		try {
			HttpRequest request = HttpRequest.newBuilder().uri(URI.create(
					"http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key=CFB8C146F9C06DC39C650F36C531D6FB&vanityurl="
							+ str))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());
			String steamId = new JSONObject(response.body()).getJSONObject("response").getString("steamid");
			HttpRequest getgames = HttpRequest.newBuilder().uri(URI.create(
					"http://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v0001/?key=CFB8C146F9C06DC39C650F36C531D6FB&steamid="
							+ steamId + "&format=json"))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> getgames2 = HttpClient.newHttpClient().send(getgames,
					HttpResponse.BodyHandlers.ofString());
			System.out.println(getgames2.body());
			return (getgames2.body());
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/steam/friendlist")
	public String getSteamFriendList(@RequestParam(value = "name", defaultValue = "VDSDU25") String str) {
		String rez = "{\"Names\":[";
		try {
			HttpRequest request = HttpRequest.newBuilder().uri(URI.create(
					"http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key=CFB8C146F9C06DC39C650F36C531D6FB&vanityurl="
							+ str))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());
			String steamId = new JSONObject(response.body()).getJSONObject("response").getString("steamid");

			HttpRequest getfriends = HttpRequest.newBuilder().uri(URI.create(
					"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=CFB8C146F9C06DC39C650F36C531D6FB&steamid="
							+ steamId + "&relationship=friend"))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> getfriends2 = HttpClient.newHttpClient().send(getfriends,
					HttpResponse.BodyHandlers.ofString());

			JSONObject list = new JSONObject(getfriends2.body());
			final JSONArray friendlist = list.getJSONObject("friendslist").getJSONArray("friends");
			final int o = friendlist.length();
			for (int i = 0; i < o; ++i) {
				final JSONObject person = friendlist.getJSONObject(i);
				HttpRequest getfriendinfo = HttpRequest.newBuilder().uri(URI.create(
						"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key=CFB8C146F9C06DC39C650F36C531D6FB&steamids="
								+ person.getString("steamid")))
						.method("GET", HttpRequest.BodyPublishers.noBody()).build();
				HttpResponse<String> getfriendinfo2 = HttpClient.newHttpClient().send(getfriendinfo,
						HttpResponse.BodyHandlers.ofString());
				JSONObject getnames = new JSONObject(getfriendinfo2.body());
				final JSONArray getnames2 = getnames.getJSONObject("response").getJSONArray("players");
				final int p = getnames2.length();
				for (int j = 0; j < p; ++j) {
					final JSONObject persona = getnames2.getJSONObject(j);
					System.out.println(persona.getString("personaname"));
					rez += "\"" + persona.getString("personaname") + "\",";
				}
			}
			rez = rez.substring(0, rez.length() - 1);
			rez += "]}";
			return rez;
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/currency")
	public String getCurrency(@RequestParam(value = "currency1", defaultValue = "EUR") String str,
			@RequestParam(value = "currency2", defaultValue = "USD") String str2,
			@RequestParam(value = "value", defaultValue = "1") String str3) {
		try {
			HttpRequest request = HttpRequest.newBuilder().uri(URI.create(
					"https://v6.exchangerate-api.com/v6/d68ab63c36e0b954f3a165e3/latest/" + str))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());

			JSONObject obj = new JSONObject(response.body());
			if (obj.getJSONObject("conversion_rates").has(str2)) {
				return (String.valueOf(Float.parseFloat(str3)
						* Float.parseFloat(obj.getJSONObject("conversion_rates").getString(str2))));
			} else
				return "No corrency with that name";
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/get/currency")
	public String getcurnames(@RequestParam(value = "currency1", defaultValue = "EUR") String str) {
		try {
			HttpRequest request = HttpRequest.newBuilder().uri(URI.create(
					"https://v6.exchangerate-api.com/v6/d68ab63c36e0b954f3a165e3/latest/" + str))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());

			JSONObject obj = new JSONObject(response.body());
			return obj.getJSONObject("conversion_rates").toString();
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/youtube/id")
	public String getYoutubebyId(@RequestParam(value = "name", defaultValue = "epitech") String str) {
		try {
			HttpRequest request = HttpRequest.newBuilder().uri(URI.create(
					"https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=" + str
							+ "&key=AIzaSyBW5_Jg9n71eD7r1RtXHodksl5YuUYWBho"))
					.method("GET", HttpRequest.BodyPublishers.noBody()).build();
			HttpResponse<String> response = HttpClient.newHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString());
			JSONObject obj = new JSONObject(response.body());
			final JSONArray obj2 = obj.getJSONArray("items");
			final int p = obj2.length();
			for (int j = 0; j < p; ++j) {
				final JSONObject person = obj2.getJSONObject(j);
				HttpRequest request2 = HttpRequest.newBuilder().uri(URI.create(
						"https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id="
								+ person.getJSONObject("snippet").getString("channelId")
								+ "&key=AIzaSyBW5_Jg9n71eD7r1RtXHodksl5YuUYWBho"))
						.method("GET", HttpRequest.BodyPublishers.noBody()).build();
				HttpResponse<String> response2 = HttpClient.newHttpClient().send(request2,
						HttpResponse.BodyHandlers.ofString());
				return (response2.body());
			}
		} catch (IOException e) {
			e.getMessage();
		} catch (InterruptedException e) {
			e.getMessage();
		}
		return "error";
	}

	@GetMapping("/about.json")
	public String about() {
		try {
			InputStream file = new FileInputStream("about.txt");
			BufferedReader buf = new BufferedReader(new InputStreamReader(file));
			String line = buf.readLine();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; line != null; ++i) {
				sb.append(line).append("\n");
				line = buf.readLine();
				if (i == 4)
					line = "        \"current_time\":" + System.currentTimeMillis() / 1000L + ",";
			}
			buf.close();
			String str = sb.toString();
			return str;
		} catch (Exception e) {
			e.getMessage();
		}
		return "error";
	}

	public static void main(String[] args) {
		SpringApplication.run(AppliApplication.class, args);
	}
}
