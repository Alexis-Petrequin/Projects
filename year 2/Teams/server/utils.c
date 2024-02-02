/*
** EPITECH PROJECT, 2021
** utils.c
** File description:
** utils functions
*/

#include "server.h"

char *get_data(int i, char *line, char *data)
{
    int y = 0;
    int a = 0;

    for (y = 0; line[y] && a < i; y++)
        if (line[y] == ';')
            a += 1;
    for (a = 0; line[y] != ';' && line[y]; a++, y++) {
        data[a] = line[y];
    }
    data[a] = '\0';
    return (data);
}

int u_help(char **command, client_t *client, server_t *srv)
{
    dprintf(client->fd, "/help:\tshow help\n/login [“user_name”]:\tset the"
    " user_name used by client\n/logout:\tdisconnect the client from the "
    "server\n/users:\tget the list of all users that exist on the domain"
    "\n/user [“user_uuid”]:\tget information about a user\n/send [“user"
    "_uuid”] [“mess age_body”]:\tsend a message to a user\n/messages "
    "[“user_uuid”]:\tlist al l messages exchange with an user\n/subscribe"
    " [“team_uuid”]:\tsubscribe  to the event of a team and its sub dire"
    "ctories (enable reception of all  events from a team)\n/subscribed ?"
    "[“team_uuid”]:\tlist all subscribed t eams or list all users subscrib"
    "ed to a team\n/unsubscribe [“team_uuid”]: \tunsubscribe from a team\n"
    "/use ?[“team_uuid”] ?[“channel_uuid”] ?[“thre ad_uuid”]:\tuse specify"
    " a context team/channel/thread\n/create:\tbased on what is being used"
    " create the sub resource\n/list:\tbased  on what is being used list "
    "all the sub resources\n/info:\tb ased on what is being used list the"
    " current\n");
    return (0);
}