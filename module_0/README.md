<h1>module_0 - Guess a number</h1>

<h3>Guess a number game</h3>
GameCore class has 4 methods with different guess number algorithm implementations.
1. game_core_v1 - random.randint based algorithm. Just a random guess.
2. game_core_v2 - this algorithm uses predicted number conditions: is predicted number greater or lower than guessed one?
3. game_core_v3 - binary search algorithm bases on random.randint
4. game_core_v4 - binary search algorithm without any random
<p>
</p>
<b>score_game(gamecore)</b> - run chosen game_core n-times in for loop. Return mean of guess a number tries.
<b>score_game_vectorize(gamecore)</b> - run chosen game_core n-times in matrix style. Reaturn meadian of guess a number tries.
<br>
<b>start_game(game_core_version=4, score_game_version=2)</b> - run chosen score_game(gamecore).
 <h2>Example</h2>
 game = GameCore()<br>
 game.start_game()
 