/*
** EPITECH PROJECT, 2021
** use.c
** File description:
** use funciton
*/

#include "server.h"

int use_return(client_t *client)
{
    if (client->place[0][0] == '\0')
        return (0);
    if (client->place[1][0] == '\0') {
        memset(client->place[0], 0, 37);
        return (0);
    }
    if (client->place[2][0] == '\0') {
        memset(client->place[1], 0, 37);
        return (0);
    }
    if (client->place[2][0] != '\0') {
        memset(client->place[2], 0, 37);
        return (0);
    }
}

int use_team(char *team_id, client_t *client)
{
    if (check_team_exist(team_id))
        return (dprintf(client->fd, "<520>\n"));
    if (check_sub_exist(team_id, client))
        return (dprintf(client->fd, "<520>\n"));
    for (int i = 0; i < 3; i++)
        memset(client->place[i], 0, 37);
    strcat(client->place[0], team_id);
    return (0);
}

int use_channel(char **command, client_t *client)
{
    if (check_team_exist(command[1]))
        return (dprintf(client->fd, "<520>\n"));
    if (check_channel_exist(command[2]))
        return (dprintf(client->fd, "<520>\n"));
    for (int i = 0; i < 3; i++)
        memset(client->place[i], 0, 37);
    strcat(client->place[0], command[1]);
    strcat(client->place[1], command[2]);
    return 0;
}

int use_thread(char **command, client_t *client)
{
    if (check_team_exist(command[1]))
        return (dprintf(client->fd, "<520>\n"));
    if (check_channel_exist(command[2]))
        return (dprintf(client->fd, "<520>\n"));
    if (check_thread_exist(command[3]))
        return (dprintf(client->fd, "<520>\n"));
    for (int i = 0; i < 3; i++)
        memset(client->place[i], 0, 37);
    strcat(client->place[0], command[1]);
    strcat(client->place[1], command[2]);
    strcat(client->place[2], command[3]);
    return (0);
}

int u_use(char **command, client_t *client, server_t *srv)
{
    if (command[1] == NULL)
        return (use_return(client));
    if (command[2] == NULL)
        return (use_team(command[1], client));
    if (command[3] == NULL)
        return (use_channel(command, client));
    if (command[3] != NULL)
        return (use_thread(command, client));
}