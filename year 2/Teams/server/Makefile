##
## EPITECH PROJECT, 2021
## makefile
## File description:
## makefile
##

SRC	=	myteams.c			\
		init_serv.c			\
		event.c 			\
		user.c				\
		createcom.c 		\
		utils.c 			\
		arg_handler.c 		\
		check_data.c 		\
		user_cmd.c 			\
		list.c 				\
		users.c 			\
		info.c  			\
		subscribed.c 		\
		use.c 				\
		messages.c 			\
		check_data2.c 		\
		arg_handler2.c 		\

OBJ	=	$(SRC:.c=.o)

NAME	=	myteams_server

all:	$(NAME)

$(NAME):	$(OBJ)
	@gcc -o $(NAME) $(OBJ) -Wall -Wextra -I../libs/myteams -L ../libs/myteams -lmyteams -luuid
	cp myteams_server ../

clean:
	@rm -f $(OBJ)

fclean: clean
	@rm -f $(NAME)
	@rm -f ../$(NAME)

re: fclean all

.PHONY: clean fclean
