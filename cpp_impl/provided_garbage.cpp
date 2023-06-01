#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

const int SIZE = 3;

void random_init(vector<vector<int>>& board) {
  srand((unsigned)time(NULL));

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      board[i][j] = 50 + (rand() % 50);
    }
  }
}

void given_init(vector<vector<int>>& board) {
  vector<vector<int>> given{{2, 4, 3}, {4, 2, 4}, {3, 4, 2}};

  board = given;
}

void print_board(vector<vector<int>>& board) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << board[i][j] << " ";
    }
    cout << endl;
  }
}

bool check_valid(vector<vector<int>> board, int row_or_col, int subtract) {
  int min = INT_MAX;
  if (subtract <= 0 || subtract > 3) {  // subtract not 1, 2 or 3
    return false;
  }
  if (row_or_col >= 0 && row_or_col < SIZE) {  // row
    for (int i = 0; i < SIZE; i++) {
      min = board[row_or_col][i] < min ? board[row_or_col][i] : min;
    }

    if (min == 0) {
      return false;
    } else if (min == 1 && subtract > 1) {
      return false;
    } else if (min == 2 && subtract > 2) {
      return false;
    }
    return true;
  } else if (row_or_col >= SIZE && row_or_col < SIZE * 2) {  // col
    row_or_col -= SIZE;
    for (int i = 0; i < SIZE; i++) {
      min = board[i][row_or_col] < min ? board[i][row_or_col] : min;
    }

    if (min == 0) {
      return false;
    } else if (min == 1 && subtract > 1) {
      return false;
    } else if (min == 2 && subtract > 2) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

void board_subtract(vector<vector<int>>& board, int row_or_col, int subtract) {
  if (row_or_col >= 0 && row_or_col < SIZE) {  // row
    for (int i = 0; i < SIZE; i++) {
      board[row_or_col][i] -= subtract;
    }
  } else if (row_or_col >= SIZE && row_or_col < SIZE * 2) {  // col
    row_or_col -= SIZE;
    for (int i = 0; i < SIZE; i++) {
      board[i][row_or_col] -= subtract;
    }
  }
}

bool check_diagonal(vector<vector<int>>& board) {
  bool diagonal1 = true;
  bool diagonal2 = true;
  for (int i = 0; i < SIZE; i++) {
    diagonal1 &= (board[i][i] == 0);
    diagonal2 &= (board[i][SIZE - i - 1] == 0);
  }
  return diagonal1 || diagonal2;
}

bool check_row(vector<vector<int>>& board,
               int row) {  // check num in row are all 0
  for (int i = 0; i < SIZE; i++) {
    if (board[row][i] != 0) {
      return false;
    }
  }
  return true;
}

bool check_col(vector<vector<int>>& board, int col) {
  for (int i = 0; i < SIZE; i++) {
    if (board[i][col] != 0) {
      return false;
    }
  }
  return true;
}

bool check_game_end(vector<vector<int>>& board, bool& dead_end) {
  bool game_end = false;
  bool row_end = true;
  bool col_end = true;

  // end condition 1 : all the numbers in any row, column, or diagonal become 0
  for (int i = 0; i < SIZE; i++) {
    game_end |= check_row(board, i);
    game_end |= check_col(board, i);
  }
  game_end |= check_diagonal(board);

  // end condition 2 : every row or column contains the number 0
  if (!game_end) {
    for (int i = 0; i < SIZE; i++) {
      bool row_exist_zero = false;
      bool col_exist_zero = false;

      for (int j = 0; j < SIZE; j++) {
        if (board[i][j] == 0) row_exist_zero = true;
        if (board[j][i] == 0) col_exist_zero = true;
      }

      if (row_exist_zero == false) row_end = false;
      if (col_exist_zero == false) col_end = false;
    }
    dead_end = row_end && col_end;
    game_end |= dead_end;
  }

  return game_end;
}

void make_your_move(vector<vector<int>> board, int& row_or_col, int& subtract) {
  //*********
  // TODO HERE:
  // 1.You CANNOT directly modify "board" in this function
  // 2.just assign "row_or_col" and "subtract" e.g. row_or_col = 3; subtract =
  // 1; 3.You can call check_valid() if you want row_or_col: 1st_row, 2nd_row,
  // 3rd_row -> 0, 1, 2 ; 1st_col, 2nd_col, 3rd_col -> 3, 4, 5 subtract: number
  // to subtract, should be 1, 2 or 3 (don't forget restriction!)
  //*********
  row_or_col = 3;
  subtract = 1;

  // cin >> row_or_col;
  // cin.get();
  // cin >> subtract;
  // cin.get();
}

void opponent_move(vector<vector<int>> board, int& row_or_col, int& subtract) {
  cin >> row_or_col;
  cin.get();

  cin >> subtract;
  cin.get();
}

int main() {
  vector<vector<int>> board(SIZE, vector<int>(SIZE));
  int player = 0;           // player 0 goes first
  int total_cost[2] = {0};  // total cost for each player
  bool your_turn = true;
  bool game_end = false;
  bool dead_end = false;
  int reward = 15;
  int penalty = 7;

  cout << "Board initialization..." << endl << endl;

  random_init(board);  // initialize board with positive integers
  // given_init(board);   // initialize board with given intergers (for testing
  // only)

  while (!game_end) {
    cout << "Current board:" << endl;
    print_board(board);
    cout << endl << "Player " << player << "'s turn:" << endl;
    system("pause");

    int row_or_col;  // row1, row2, row3 -> 0, 1, 2 ; col1, col2, col3 -> 3, 4,
                     // 5
    int subtract;    // number to subtract
    bool valid;      // legal move checking

    if (your_turn) {
      time_t start, end, diff;
      start = time(nullptr);
      make_your_move(board, row_or_col, subtract);
      end = time(nullptr);

      cout << "Time: " << difftime(end, start) << " seconds" << endl << endl;
    } else {
      // time_t start, end, diff;
      // start = time(nullptr);
      opponent_move(board, row_or_col, subtract);
      // end = time(nullptr);

      // cout << "Time: " << difftime(end, start) << " seconds" << endl << endl;
    }

    cout << "Player " << player << "'s move: ";
    cout << "row_or_col: " << row_or_col << " subtract: " << subtract << endl
         << endl;

    cout << "Valid checking...";
    valid = check_valid(board, row_or_col, subtract);
    if (valid)
      cout << "The move is vaild." << endl;
    else
      cout << "The move is invalid, game over." << endl;
    system("pause");

    board_subtract(board, row_or_col, subtract);

    // update player's total cost
    total_cost[player] += subtract;
    cout << "Player 0 total cost: " << total_cost[0] << endl;
    cout << "Player 1 total cost: " << total_cost[1] << endl;
    cout << "--------------------------------------" << endl;

    // check if game has ended
    game_end = check_game_end(board, dead_end);

    if (!game_end) {
      // switch to other player
      your_turn = !your_turn;
      player = (player + 1) % 2;
    }
  }

  // print final board and result
  cout << "Final board:" << endl;
  print_board(board);
  if (!dead_end) {
    cout << "Player " << player << " ends with a diagonal/row/col of 0's!"
         << endl;
    total_cost[player] -= reward;
  } else {
    cout << "Player " << player << " ends with a dead end!" << endl;
    total_cost[player] += penalty;
  }

  cout << "Player 0 total cost: " << total_cost[0] << endl;
  cout << "Player 1 total cost: " << total_cost[1] << endl;
  system("pause");

  return 0;
}