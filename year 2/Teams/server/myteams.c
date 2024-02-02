/*
** EPITECH PROJECT, 2021
** myteams.c
** File description:
** myteams functions
*/

#include "server.h"

int do_command(char **command, client_t *client, server_t *srv)
{
    char function[15][20] = {"/login", "/help", "/logout", "/users",
    "/messages", "/subscribe", "/subscribed", "/unsubscribe", "/use",
    "/list", "/info", "/create", "/user", "/send" };
    int (*fptr[15])(char **command, client_t *client, server_t *srv) =
    { u_login, u_help, u_logout, u_users, u_messages, u_sub, u_subscribed,
    u_unsub, u_use, u_list, u_info, u_create, u_user, u_send };

    for (int i = 0; i < 15; i++)
        if (strcmp(command[0], function[i]) == 0) {
            if (client->logged_in == false && i > 0)
                return (dprintf(client->fd, "<520>\n"));
            return (fptr[i](command, client, srv));
        }
    return (dprintf(client->fd, "<520>\n"));
}

int handle_msg(client_t *client, server_t *srv)
{
    char **map;
    int nb = 0;

    nb = parse_buffer(client->buffer, &map);
    if (nb < 0 || check_args_function(map[0], nb)) {
        if (nb == -2) {
            for (int i = 0; map[i]; i++)
                free(map[i]);
            free(map);
        }
        dprintf(client->fd, "<520>\n");
        return (84);
    }
    do_command(map, client, srv);
    for (int i = 0; map[i]; i++)
        free(map[i]);
    free(map);
    return (0);
}

int handle_client(server_t *srv, int i)
{
    int valread = 0;
    int sd = 0;
    char buff[MAX_CMD + 1] = { 0 };

    if (FD_ISSET(srv->client[i].fd, srv->fd_read)) {
        valread = read(srv->client[i].fd, buff, 1024);
        if (valread == 0)
            connection_lost(srv, i);
        if (valread > 1 && buff[0] != '\r' && buff[0] != 0)
            if (handle_buffer(buff, &srv->client[i]) == 0)
                return (handle_msg(&srv->client[i], srv));
    }
    return (0);
}

void initloop(server_t *server)
{
    FD_ZERO(server->fd_read);
    FD_ZERO(server->fd_write);
    FD_SET(server->fd_srv, server->fd_read);
    server->max_sd = server->fd_srv;
    for (int i = 0; i < MAX_CLIENT; ++i) {
        if (server->client[i].fd > 0) {
            FD_SET(server->client[i].fd, server->fd_read);
            FD_SET(server->client[i].fd, server->fd_write);
            memset(server->client[i].buffer, 0, MAX_CMD + 1);
        }
        if (server->client[i].fd > server-> max_sd)
            server->max_sd = server->client[i].fd;
    }
}

int main(int argc, char** argv)
{
    server_t *server = NULL;

    if (check_errs(argc, argv))
        return (84);
    server = init_serv(atoi(argv[1]));
    if (creat_srv(server))
        return (84);
    while (1) {
        initloop(server);
        select(server->max_sd + 1, server->fd_read,
        server->fd_write, NULL, NULL);
        if (FD_ISSET(server->fd_srv, server->fd_read))
            new_connection(server);
        for (int i = 0; i < MAX_CLIENT; ++i)
            handle_client(server, i);
    }
    return (0);
}