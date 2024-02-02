# Teams
Second-year project, a server able to manage a collaborative communication application like the well known Microsoft Teams ®

## Features

A collaborative communication application is a service able to manage severall communication teams, where discussion are organised like following:
* threads (initial post and additional comments) in a specific channel
* discussion (personnal messages)

Severals actions available:
* Creating/Joining/Leaving a team
* Creating a user
* Creating a channel in a team
* Creating a thread in a channel
* Creating a comment in a thread
* Saving & restoring users, teams, channels, threads & associated comments
* Personnal discussion (from a user to an other)
* Saving & restoring personnal discussion

### Client

Client will handle the following command from the standard input :
* /help : show help
* /login [“user_name”] : set the user_name used by client
* /logout : disconnect the client from the server
* /users : get the list of all users that exist on the domain
* /user [“user_uuid”] : get information about a user
* /send [“user_uuid”] [“message_body”] : send a message to a user
* /messages [“user_uuid”] : list all messages exchange with an user
* /subscribe [“team_uuid”] : subscribe to the event of a team and its sub directories (enable reception of all events from a team)
* /subscribed ?[“team_uuid”] : list all subscribed teams or list all users subscribed to a team
* /unsubscribe [“team_uuid”] : unsubscribe from a team
* /use ?[“team_uuid”] ?[“channel_uuid”] ?[“thread_uuid”] : use specify a context team/channel/thread
* /create : based on what is being used create the sub resource
* /list : based on what is being used list all the sub resources
* /info : based on what is being used list the current

#### /CREATE

When the context is not defined (/use):
* /create [“team_name”] [“team_description”] : create a new team
When team_uuid is defined (/use “team_uuid”):
* /create [“channel_name”] [“channel_description”] : create a new channel
When team_uuid and channel_uuid are defined (/use “team_uuid” “channel_uuid”):
* /create [“thread_title”] [“thread_message”] : create a new thread
When team_uuid, channel_uuid and thread_uuid are defined (/use “team_uuid” “channel_uuid” “thread_uuid”):
* /create [“comment_body”] : create a new reply

#### /LIST
When the context is not defined (/use):
* /list : list all existing teams
When team_uuid is defined (/use “team_uuid”):
* /list : list all existing channels
When team_uuid and channel_uuid are defined (/use “team_uuid” “channel_uuid”):
* /list : list all existing threads
When team_uuid, channel_uuid and thread_uuid are defined (/use “team_uuid” “channel_uuid” “thread_uuid”):
* /list : list all existing replies

#### /INFO
When the context is not defined (/use):
* /info : display currently logged user infos
When team_uuid is defined (/use “team_uuid”):
* /info : display currently selected team infos
When team_uuid and channel_uuid are defined (/use “team_uuid” “channel_uuid”):
* /info : display currently selected channel infos
When team_uuid, channel_uuid and thread_uuid are defined (/use “team_uuid” “channel_uuid” “thread_uuid”):
* /info : display currently selected thread infos

#### GENERAL INFORMATIONS
Please note that all arguments of the existing commands should be quoted with double quotes.
A missing quote should be interpreted as an error.
Please note that all the names, descriptions and message bodies have a pre-defined length which will be
as follow:
* MAX_NAME_LENGTH 32
* MAX_DESCRIPTION_LENGTH 255
* MAX_BODY_LENGTH 512

## Build with linux (ubuntu)

### 1. Set environement variable (for each terminal):

	export LD_LIBRARY_PATH=/Path/to/folder/Teams/libs/myteams

### 2. Build teams from my repo:
	make
	./myteams_server [PORT] (ex: 4000)
    ./myteams_cli [I.P.] [Port used by server] (ex: 127.0.0.1 4000)
