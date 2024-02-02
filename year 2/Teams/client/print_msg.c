/*
** EPITECH PROJECT, 2021
** print_msg.c
** File description:
** print server response
*/

#include "client.h"

int print_message4(char **args)
{
    if (strcmp(args[0], "413") == 0)
        return (client_print_reply_created(args[1], args[2], atoi(args[3]),
        args[4]));
    if (strcmp(args[0], "420") == 0)
        return (client_print_subscribed(args[1], args[2]));
    if (strcmp(args[0], "430") == 0)
        return (client_print_unsubscribed(args[1], args[2]));
    return (0);
}

int print_message3(char **args)
{
    if (strcmp(args[0], "520") == 0)
        return (client_error_unauthorized());
    if (strcmp(args[0], "530") == 0)
        return (client_error_already_exist());
    if (strcmp(args[0], "310") == 0)
        return (client_print_user(args[1], args[2], atoi(args[3])));
    if (strcmp(args[0], "311") == 0)
        return (client_print_team(args[1], args[2], args[3]));
    if (strcmp(args[0], "312") == 0)
        return (client_print_channel(args[1], args[2], args[3]));
    if (strcmp(args[0], "313") == 0)
        return (client_print_thread(args[1], args[2], atoi(args[3]), args[4],
        args[5]));
    if (strcmp(args[0], "410") == 0)
        return (client_print_team_created(args[1], args[2], args[3]));
    if (strcmp(args[0], "411") == 0)
        return (client_print_channel_created(args[1], args[2], args[3]));
    if (strcmp(args[0], "412") == 0)
        return (client_print_thread_created(args[1], args[2], atoi(args[3]),
        args[4], args[5]));
    return (print_message4(args));
}

int print_message2(char **args)
{
    if (strcmp(args[0], "212") == 0)
        return (client_team_print_channels(args[1], args[2], args[3]));
    if (strcmp(args[0], "213") == 0)
        return (client_channel_print_threads(args[1], args[2], atoi(args[3]),
        args[4], args[5]));
    if (strcmp(args[0], "214") == 0)
        return (client_thread_print_replies(args[1], args[2], atoi(args[3]),
        args[4]));
    if (strcmp(args[0], "215") == 0)
        return (client_private_message_print_messages(args[1], atoi(args[2]),
        args[3]));
    if (strcmp(args[0], "510") == 0)
        return (client_error_unknown_team(args[1]));
    if (strcmp(args[0], "511") == 0)
        return (client_error_unknown_channel(args[1]));
    if (strcmp(args[0], "512") == 0)
        return (client_error_unknown_thread(args[1]));
    if (strcmp(args[0], "513") == 0)
        return (client_error_unknown_user(args[1]));
    return (print_message3(args));
}

int print_message(char **args)
{
    if (strcmp(args[0], "110") == 0)
        return (client_event_logged_in(args[1], args[2]));
    if (strcmp(args[0], "120") == 0)
        return (client_event_logged_out(args[1], args[2]));
    if (strcmp(args[0], "130") == 0)
        return (client_event_private_message_received(args[1], args[2]));
    if (strcmp(args[0], "143") == 0)
        return (client_event_thread_reply_received(args[1], args[2],
        args[3], args[4]));
    if (strcmp(args[0], "140") == 0)
        return (client_event_team_created(args[1], args[2], args[3]));
    if (strcmp(args[0], "141") == 0)
        return (client_event_channel_created(args[1], args[2], args[3]));
    if (strcmp(args[0], "142") == 0)
        return (client_event_thread_created(args[1], args[2],
        atoi(args[3]), args[4], args[5]));
    if (strcmp(args[0], "210") == 0)
        return (client_print_users(args[1], args[2], atoi(args[3])));
    if (strcmp(args[0], "211") == 0)
        return (client_print_teams(args[1], args[2], args[3]));
    return (print_message2(args));
}