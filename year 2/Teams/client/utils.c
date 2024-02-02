/*
** EPITECH PROJECT, 2021
** client
** File description:
** client function
*/

#include "client.h"

int check_errors(int argc, char **argv)
{
    if (argc != 3)
        return (84);
    for (int i = 0; argv[1][i]; ++i)
        if ((argv[1][i] < '0' || argv[1][i] > '9') && argv[1][i] != '.')
            return (84);
    for (int i = 0; argv[2][i]; ++i)
        if (argv[2][i] < '0' || argv[2][i] > '9')
            return (84);
    return 0;
}

int end(client_t *client, int ret)
{
    close(client->fd);
    free(client);
    return ret;
}

int get_nbarg(char *buff)
{
    int nb = 0;

    for (int i = 0; buff[i]; i++)
        if (buff[i] == '<')
            nb++;
    return (nb);
}

char **get_arg(char *buff)
{
    int nb = get_nbarg(buff);
    char **args = malloc(sizeof(char *) * (nb + 1));
    int y = 0;
    int n = 0;

    for (int i = 0; i < nb; i++){
        args[i] = malloc(MAX_COM);
        memset(args[i], 0, MAX_COM);
    }
    args[nb] = NULL;
    for (int i = 0; i < nb; i++) {
        for (;buff[y] == '<' || buff[y] == '>'; y++);
        for (n = 0;buff[y] != '>' && buff[y]; y++, n++)
            args[i][n] = buff[y];
    }
    return (args);
}