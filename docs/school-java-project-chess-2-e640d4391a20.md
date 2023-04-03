# 学校 Java 项目国际象棋(2)

> 原文：<https://medium.com/analytics-vidhya/school-java-project-chess-2-e640d4391a20?source=collection_archive---------8----------------------->

![](img/3f1801577ea74c3888ca6fb5931f2e3e.png)

## 部署 32 个初始部件

上次我们打印出了空棋盘。今天，我们将用最初的 32 件物品填充白板。当我们完成时，板子看起来会像下面这样。

```
 a b c d e f g h
8 r n b q k b n r 8
7 p p p p p p p p 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 P P P P P P P P 2
1 R N B Q K B N R 1
  a b c d e f g h
```

打开 Chess.java 并在文件末尾创建一个枚举`Type`来表示棋子类型。

```
class Chess {
  public static void main(String[] args) {
    Board brd = new Board();
    System.out.println(brd);
  }
}class Board {
  public String toString() {
    String brdStr = "";
    brdStr += "  0 1 2 3 4 5 6 7\n";
    for (int row = 0; row < 8; row++) {
      brdStr += row + "";
      for (int col = 0; col < 8; col++) {
        brdStr += " .";
      }
      brdStr += "\n"; // line break
    }
    return brdStr;
  }
}**enum Type {
  P, // pawn
  R, // rook
  N, // knight
  B, // bishop
  Q, // queen
  K, // king
}**
```

编辑/编译/运行 Chess.java。

```
🄹 vim Chess.java                
🄹 javac Chess.java && java Chess
  0 1 2 3 4 5 6 7
0 . . . . . . . .
1 . . . . . . . .
2 . . . . . . . .
3 . . . . . . . .
4 . . . . . . . .
5 . . . . . . . .
6 . . . . . . . .
7 . . . . . . . .
```

现在我们可以设计一个新的类`Piece`来表示一块。将以下代码添加到文件 Chess.java 的末尾。

```
class Piece {
  int c; // column
  int r; // row
  boolean w; // isWhite
  Type t;Piece(int c, int r, boolean w, Type t) {
    this.c = c;
    this.r = r;
    this.w = w;
    this.t = t;
  }
}
```

我们可以使用两个 vim 命令来稍微加快编辑速度。“L”将光标移动到底部。“o”进入插入模式，并在光标下方插入一个新行。顺便说一下，“H”移动光标到顶部，“M”移动到中间。“O”进入插入模式并在光标上方插入一个新行。

编辑/编译/运行 Chess.java 以确保一切正常。

```
🄹 vim Chess.java                
🄹 javac Chess.java && java Chess
  0 1 2 3 4 5 6 7
0 . . . . . . . .
1 . . . . . . . .
2 . . . . . . . .
3 . . . . . . . .
4 . . . . . . . .
5 . . . . . . . .
6 . . . . . . . .
7 . . . . . . . .
```

为了使用数据类型集作为所有组件的容器，我们需要在 Chess.java 的顶部添加两行代码。

```
import java.util.Set;
import java.util.HashSet;
```

我们将创建一个类型设置为<piece>的实例变量，并在构造函数`Board()`中插入一个白皇后。</piece>

```
class Board {
  **private Set<Piece> pieces = new HashSet<>();** **Board() {
    pieces.add(new Piece(3, 7, true, Type.Q));
  }** public String toString() {
    String brdStr = "";
    brdStr += "  0 1 2 3 4 5 6 7\n";
    for (int row = 0; row < 8; row++) {
      brdStr += row + "";
      for (int col = 0; col < 8; col++) {
        brdStr += " .";
      }
      brdStr += "\n"; // line break
    }
    return brdStr;
  }
}
```

通过编译和运行更新的 Chess.java 来确保它仍然工作。

为了在板上打印出 Piece，我们可以创建下面的 helper 方法`Piece pieceAt(int c, int r)`来返回一个特定位置的 Piece 对象。

```
class Board {
  private Set<Piece> pieces = new HashSet<>(); Board() {
    pieces.add(new Piece(3, 7, true, Type.Q));
  } **Piece pieceAt(int c, int r) {
    for (Piece p : pieces) {
      if (p.c == c && p.r == r) {
        return p;
      }
    }
    return null;
  }** public String toString() {
    String brdStr = "";
    brdStr += "  0 1 2 3 4 5 6 7\n";
    for (int row = 0; row < 8; row++) {
      brdStr += row + "";
      for (int col = 0; col < 8; col++) {
        brdStr += " .";
      }
      brdStr += "\n"; // line break
    }
    return brdStr;
  }
}
```

现在我们需要修改类 Board 的 toString()方法来显示片段。

```
class Board {
  private Set<Piece> pieces = new HashSet<>(); Board() {
    pieces.add(new Piece(3, 7, true, Type.Q));
  } Piece pieceAt(int c, int r) {
    for (Piece p : pieces) {
      if (p.c == c && p.r == r) {
        return p;
      }
    }
    return null;
  } public String toString() {
    String brdStr = "";
    brdStr += "  0 1 2 3 4 5 6 7\n";
    for (int r = 0; r < 8; r++) {
      brdStr += r + "";
      for (int c = 0; c < 8; c++) {
        Piece p = pieceAt(c, r);
        if (p == null) {
          brdStr += " .";
        } else {
          switch (p.t) {
          case P: brdStr += p.w ? " P" : " p"; break;
          case R: brdStr += p.w ? " R" : " r"; break;
          case N: brdStr += p.w ? " N" : " n"; break;
          case B: brdStr += p.w ? " B" : " b"; break;
          case Q: brdStr += p.w ? " Q" : " q"; break;
          case K: brdStr += p.w ? " K" : " k"; break;
          }
        }
      }
      brdStr += "\n"; // line break
    }
    return brdStr;
  }
}
```

编辑/编译/运行 Chess.java，看看船上的单件女王。

```
🄹 vim Chess.java                
🄹 javac Chess.java && java Chess
  a b c d e f g h
8 . . . . . . . . 8
7 . . . . . . . . 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 . . . . . . . . 2
1 . . . Q . . . . 1
  a b c d e f g h
```

是时候在`Board`类的构造函数中添加剩余的 31 个代码了。

```
 Board() {
    pieces.add(new Piece(3, 0, false, Type.Q));
    pieces.add(new Piece(3, 7, true , Type.Q));
    pieces.add(new Piece(4, 0, false, Type.K));
    pieces.add(new Piece(4, 7, true , Type.K));
    for (int i = 0; i < 2; i++) {
      pieces.add(new Piece(0 + i * 7, 0, false, Type.R));
      pieces.add(new Piece(0 + i * 7, 7, true , Type.R));
      pieces.add(new Piece(1 + i * 5, 0, false, Type.N));
      pieces.add(new Piece(1 + i * 5, 7, true , Type.N));
      pieces.add(new Piece(2 + i * 3, 0, false, Type.B));
      pieces.add(new Piece(2 + i * 3, 7, true , Type.B));
    }
    for (int i = 0; i < 8; i++) {
      pieces.add(new Piece(i, 1, false, Type.P));
      pieces.add(new Piece(i, 6, true , Type.P));
    }
  }
```

编辑/编译/运行 Chess.java 看到所有 32 件。

```
🄹 vim Chess.java                
🄹 javac Chess.java && java Chess
  a b c d e f g h
8 r n b q k b n r 8
7 p p p p p p p p 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 P P P P P P P P 2
1 R N B Q K B N R 1
  a b c d e f g h
```

这是目前为止完整的代码。

```
import java.util.Set;
import java.util.HashSet;class Chess {
  public static void main(String[] args) {
    Board brd = new Board();
    System.out.println(brd);
  }
}class Board {
  private Set<Piece> pieces = new HashSet<>(); Board() {
    pieces.add(new Piece(3, 0, false, Type.Q));
    pieces.add(new Piece(3, 7, true , Type.Q));
    pieces.add(new Piece(4, 0, false, Type.K));
    pieces.add(new Piece(4, 7, true , Type.K));
    for (int i = 0; i < 2; i++) {
      pieces.add(new Piece(0 + i * 7, 0, false, Type.R));
      pieces.add(new Piece(0 + i * 7, 7, true , Type.R));
      pieces.add(new Piece(1 + i * 5, 0, false, Type.N));
      pieces.add(new Piece(1 + i * 5, 7, true , Type.N));
      pieces.add(new Piece(2 + i * 3, 0, false, Type.B));
      pieces.add(new Piece(2 + i * 3, 7, true , Type.B));
    }
    for (int i = 0; i < 8; i++) {
      pieces.add(new Piece(i, 1, false, Type.P));
      pieces.add(new Piece(i, 6, true , Type.P));
    }
  } Piece pieceAt(int c, int r) {
    for (Piece p : pieces) {
      if (p.c == c && p.r == r) {
        return p;
      }
    }
    return null;
  } public String toString() {
    String brdStr = "";
    brdStr += "  a b c d e f g h\n";
    for (int r = 0; r < 8; r++) {
      brdStr += (8 - r) + "";
      for (int c = 0; c < 8; c++) {
        Piece p = pieceAt(c, r);
        if (p == null) {
          brdStr += " .";
        } else {
          switch (p.t) {
          case P: brdStr += p.w ? " P" : " p"; break;
          case R: brdStr += p.w ? " R" : " r"; break;
          case N: brdStr += p.w ? " N" : " n"; break;
          case B: brdStr += p.w ? " B" : " b"; break;
          case Q: brdStr += p.w ? " Q" : " q"; break;
          case K: brdStr += p.w ? " K" : " k"; break;
          }
        }
      }
      brdStr += " " + (8 - r) + "\n";
    }
    brdStr += "  a b c d e f g h\n";
    return brdStr;
  }
}enum Type {
  P, // pawn
  R, // rook
  N, // knight
  B, // bishop
  Q, // queen
  K, // king
}class Piece {
  int c; // column
  int r; // row
  boolean w; // isWhite
  Type t;Piece(int c, int r, boolean w, Type t) {
    this.c = c;
    this.r = r;
    this.w = w;
    this.t = t;
  }
}
```

我们结束了。

[学校 Java 项目象棋(1)](/@zhijunsheng/school-java-project-chess-1-85f97a2d1877?source=friends_link&sk=eab9b54731072ce1aefe26f2bf2f2d7b)

[学校 Java 项目象棋(2)](/@zhijunsheng/school-java-project-chess-2-e640d4391a20?source=friends_link&sk=502668bdbc9ff9291585144d5702c2fc)