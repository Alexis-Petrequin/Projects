/*
** EPITECH PROJECT, 2021
** user_cmd.c
** File description:
** user commands
*/

#include "server.h"

int add_sub(client_t *client, char *uuid)
{
    char str[50] = { "Database/"};
    FILE *fd = fopen(strcat(str, client->id), "a+");
    FILE *fd2 = fopen(PATH_TM, "r");
    char name[MAX_NAME + 1] = { 0 };
    char desc[MAX_DESCRIPTION + 1] = { 0 };
    char *line = NULL;
    char id[37] = { 0 };
    size_t len = 0;

    if (fd2 == NULL) {
        fclose(fd);
        return (84);
    }
    while (getline(&line, &len, fd2) != -1)
        if (strcmp(uuid, get_data(1, line, id)) == 0)
            break;
    fprintf(fd, "%s;team;%s;%s;\n", uuid, get_data(0, line, name),
    get_data(3, line, desc));
    fclose(fd2);
    fclose(fd);
    return (0);
}

int u_sub(char **command, client_t *client, server_t *srv)
{
    if (check_team_exist(command[1]))
        return (dprintf(client->fd, "<510><%s>\n", command[1]));
    if (check_sub_exist(command[1], client) == 0)
        return (dprintf(client->fd, "<520>\n"));
    if (!add_sub(client, command[1]))
        dprintf(client->fd, "<420><%s><%s>\n", client->id, command[1]);
    server_event_user_subscribed(command[1], client->id);
    return (0);
}

int del_sub(client_t *client, char *team_uuid)
{
    char str[50] = { "Database/"};
    FILE *fd = fopen(strcat(str, client->id), "r+");
    char *id = malloc(37);
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int s = 0;

    for (s = 0; (read = getline(&line, &len, fd)) != -1; s += read)
        if (strcmp(get_data(0, line, id), team_uuid) == 0)
                break;
    fseek(fd, s, SEEK_SET);
    for (int i = 0; i < 36; i++)
        fprintf(fd, "0");
    free(id);
    fclose(fd);
    return (0);
}

int u_unsub(char **command, client_t *client, server_t *srv)
{
    if (check_team_exist(command[1]))
        return (dprintf(client->fd, "<510><%s>\n", command[1]));
    if (check_sub_exist(command[1], client))
        return (dprintf(client->fd, "<520>\n"));
    del_sub(client, command[1]);
    dprintf(client->fd, "<430><%s><%s>\n", client->id, command[1]);
    server_event_user_unsubscribed(command[1], client->id);
    return (0);
}