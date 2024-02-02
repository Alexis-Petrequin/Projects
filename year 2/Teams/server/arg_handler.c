/*
** EPITECH PROJECT, 2021
** arg_handler.c
** File description:
** abouter handling args
*/

#include "server.h"

int pre_parse2(char ***map2, int count, int y)
{
    if (count % 2 != 0 || y == -1)
        return (-1);
    count /= 2;
    map2[0] = malloc(sizeof(char*) * (count + 2));
    for (int f = 0; f < count + 1; ++f) {
        map2[0][f] = malloc(MAX_CMD + 1);
        memset(map2[0][f], 0, MAX_CMD + 1);
    }
    map2[0][count + 1] = NULL;
    return (count);
}

int pre_parse(char *str, char ***map2)
{
    int count = 0;
    int y = 0;
    int t = 0;

    for (t = 0; str[t] && str[t] != ' '; t++);
    for (; str[t] && (str[t] == '/' || (str[t] >= 'a' && str[t] <= 'z') ||
    (str[t] >= 'A' && str[t] <= 'Z')); t++);
    for (; str[t] && str[t] != '\"'; t++)
        if ((str[t] != '\n' && str[t] != ' ') || str[0] == '\"')
            y = -1;
    for (int i = 0; str[i] != '\0'; ++i) {
        if (str[i] == '\"')
            ++count;
        if (count % 2 == 0 && (str[i] != ' ' && str[i] != '\"' &&
        str[i] != '\n') && count > 0)
            y = -1;
    }
    return (pre_parse2(map2, count, y));
}

int do_parse(char *str, char ***map, int *c, int *a)
{
    int y = 0;

    for (int b = 0; str[c[0]] != '\0' && y < 2; c[0]++) {
        if (y == 0 && str[c[0]] != ' ' && str[c[0]] != '\"') {
            c[0] = -1;
            return (-2);
        }
        if (y == 1 && str[c[0]] != '\"') {
            map[0][a[0]][b] = str[c[0]];
            b++;
        }
        if (str[c[0]] == '\"')
            y += 1;
    }
    return (0);
}

int parse_buffer(char *str, char ***map)
{
    int count = pre_parse(str, map);
    int y = 0;
    int c = 0;

    if (count == -1)
        return (-1);
    for (c = 0; str[c] && str[c] == ' '; c++);
    for (int i = 0; str[c] && str[c] != ' ' && str[c] != '\n'; c++, i++)
        map[0][0][i] = str[c];
    if (c == strlen(str))
        return (1);
    for (int a = 1; a < (count + 1); a++)
        if (do_parse(str, map, &c, &a) == -2)
            break;
    if (c < 0) {
        for (int i = 0; i < count + 1; i++)
            free(map[0][i]);
        free (map[0]);
        return (-2);
    }
    return (count + 1);
}

int check_args_function(char *buffer, int nb)
{
    char commands[19][20] = {"/help", "/login", "/logout", "/users", "/user",
    "/send", "/messages", "/subscribe", "/subscribed", "/subscribed",
    "/unsubscribe", "/use", "/use", "/use", "/use", "/list","/info",
    "/create", "/create" };
    int nbarg[] = {1, 2, 1, 1, 2, 3, 2, 2, 1, 2, 2, 2, 1, 3, 4, 1, 1, 2, 3};

    for (int i = 0; i < 19; i++)
        if (strcmp(commands[i], buffer) == 0 && nb == nbarg[i])
            return (0);
    return (84);
}