##
## EPITECH PROJECT, 2019
## Makefile
## File description:
## makefile
##

SRC	= 	my_getnbr.c			\
		my_pool.c			\
		menu.c				\
		menu_init.c			\
		menu_init2.c			\
		button.c			\
		battle_init.c			\
		battle_init2.c			\
		battle_draw.c			\
		init_other.c			\
		utils.c				\
		battle_attacks.c		\
		battle_fire.c			\
		battle_items.c			\
		particules.c			\
		battle_ia.c			\
		battle_end.c			\
		map_move.c			\
		map_checkborder.c		\
		map_interaction.c		\
		map_interface.c			\
		map.c				\
		map_init_sprite.c		\
		map_init.c			\
		map_draw_all.c			\
		map_draw_all2.c			\
		map_event.c			\
		map_button.c			\
		map_update.c			\
		free.c				\
		free2.c				\
		battle.c			\
		battle_event.c			\
		battle_draw2.c			\

OBJ	=	$(SRC:.c=.o)

NAME	=	libmy.a

all:	$(NAME)

$(NAME):	$(OBJ)
	ar rc $(NAME) $(OBJ)

clean:
	rm -f $(OBJ)

fclean:	clean
	rm -f $(NAME)

re: fclean all
