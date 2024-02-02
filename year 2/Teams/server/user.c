/*
** EPITECH PROJECT, 2021
** user.c
** File description:
** user create
*/

#include "server.h"

int available_name(char *name)
{
    FILE *fd = fopen(PATH_US, "a+");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char temp[MAX_NAME + 3] = { 0 };

    strcat(temp, name);
    strcat(temp, ";\n");
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(temp, line + 37) == 0) {
            fclose(fd);
            return (84);
        }
    fclose(fd);
    return (0);
}

int create_user(char *name, client_t *client)
{
    FILE *fd = fopen(PATH_US, "a+");
    uuid_t id;
    char t_uuid[37] = { 0 };

    if (strlen(name) > MAX_NAME)
        return (84);
    uuid_generate_random(id);
    uuid_unparse_upper(id, client->id);
    strcat(client->pseudo, name);
    fprintf(fd, "%s;%s;\n", client->id, name);
    client->logged_in = true;
    memset(client->pseudo, 0, MAX_NAME + 1);
    strcat(client->pseudo, name);
    server_event_user_created(client->id, name);
    fclose(fd);
    server_event_user_logged_in(client->id);
    return (0);
}

int user_loging(char *name, client_t *client)
{
    FILE *fd = fopen(PATH_US, "a+");
    char *line = NULL;
    size_t len = 0;
    char temp[MAX_NAME + 3] = { 0 };
    char temp2[37] = { 0 };

    strcat(temp, name);
    strcat(temp, ";\n");
    while (getline(&line, &len, fd) != -1)
        if (strcmp(temp, line + 37) == 0) {
            client->logged_in = true;
            memset(client->pseudo, 0, MAX_NAME + 1);
            strcat(client->pseudo, name);
            memset(client->id, 0, 37);
            strcat(client->id, get_data(0, line, temp2));
            break;
        }
    server_event_user_logged_in(client->id);
    fclose(fd);
    return (0);
}

int u_login(char **command, client_t *client, server_t *srv)
{
    if (client->logged_in == true) {
        dprintf(client->fd, "<520>\n");
        return (84);
    }
    if (available_name(command[1]) == 0) {
        create_user(command[1], client);
        for (int i = 0; srv->client[i].fd > 0; i++)
            dprintf(srv->client[i].fd, "<110><%s><%s>\n",
            client->id, client->pseudo);
    }
    else {
        user_loging(command[1], client);
        for (int i = 0; srv->client[i].fd > 0; i++)
            dprintf(srv->client[i].fd, "<110><%s><%s>\n",
            client->id, client->pseudo);
    }
    return (0);
}

void user_detect(void)
{
    FILE *fd = fopen(PATH_US, "r");
    char *li = NULL;
    size_t len = 0;
    ssize_t read;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };

    while ((read = getline(&li, &len, fd)) != -1)
        server_event_user_loaded(get_data(0, li, id), get_data(1, li, name));
    fclose(fd);
}