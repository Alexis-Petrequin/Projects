/*
** EPITECH PROJECT, 2021
** subscribed.c
** File description:
** function subscribed
*/

#include "server.h"

int list_sub(client_t *client)
{
    char str[50] = { "Database/" };
    FILE *fd = fopen(strcat(str, client->id), "r");
    char id[37] = { 0 };
    char id2[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char desc[MAX_DESCRIPTION + 1] = { 0 };
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp("team", get_data(1, line, id2)) == 0 && line[8] == '-')
            dprintf(client->fd, "<211><%s><%s><%s>\n",get_data(0, line, id),
            get_data(2, line, name), get_data(3, line, desc));
    fclose(fd);
    return (0);
}

int list_teamer2(char *team_id, client_t *client, char *line, server_t *srv)
{
    char id[37] = { 0 };
    char id2[37] = { 0 };
    char str[50] = { "Database/"};
    char name[MAX_NAME + 1] = { 0 };
    size_t len = 0;
    char *li = NULL;
    FILE *fd = fopen(strcat(str, get_data(0, line, id2)), "r");

    if (fd == NULL)
        return (84);
    while (getline(&li, &len, fd) != -1)
        if (strcmp(team_id, get_data(0, li, id)) == 0)
            dprintf(client->fd, "<210><%s><%s><%c>\n",
            id2, get_data(1, line, name), user_status(srv, id2));
    fclose(fd);
    return (0);
}

int list_teamer(char *team_id, client_t *client, server_t *srv)
{
    FILE *fd = fopen(PATH_US, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    while (getline(&li, &len, fd) != -1)
        list_teamer2(team_id, client, li, srv);
    fclose(fd);
    return (0);
}

int u_subscribed(char **command, client_t *client, server_t *srv)
{
    if (command[1] == NULL)
        return (list_sub(client));
    if (check_team_exist(command[1]))
        return (dprintf(client->fd, "<510><%s>\n", command[1]));
    return (list_teamer(command[1], client, srv));
}