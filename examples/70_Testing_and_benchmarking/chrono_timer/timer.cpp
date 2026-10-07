#include <chrono>
#include <cstdio>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::steady_clock;
using std::chrono::time_point;

class MarkTimer {
public:
  int mark() {
    marks.push_back(steady_clock::now());
    return marks.size() - 1;
  }

  double timeSince(int mark) {
    return duration_cast<std::chrono::milliseconds>(steady_clock::now() -
                                                    marks[mark])
        .count();
  }

  double timeBetween(int start, int end) {
    return duration_cast<std::chrono::milliseconds>(marks[end] - marks[start])
        .count();
  }

private:
  std::vector<std::chrono::time_point<steady_clock>> marks = {};
};

template <typename TUnit> class LapTimer {
public:
  LapTimer() { curr = now(); }

  static double now() {
    return duration_cast<TUnit>(steady_clock::now().time_since_epoch()).count();
  }

  void reset() { curr = now(); }

  double lap() {
    double diff = now() - curr;
    reset();
    return diff;
  }

private:
  double curr;
};

int main() {
  {
    Timer timer;

    int initTimerMark = timer.mark();

    for (int i = 0; i < 1000000000; i++) {
      int j = i + 10000;
    }

    int finalTimerMark = timer.mark();

    printf("Time since beginning (ms): %f\n", timer.timeSince(initTimerMark));
    printf("Time in loop (ms): %f\n",
           timer.timeBetween(initTimerMark, finalTimerMark));
  }

  {
    LapTimer<std::chrono::milliseconds> timer;

    for (int i = 0; i < 1000000000; i++) {
      int j = i + 10000;
    }

    printf("Time in loop (ms): %f\n", timer.lap());
  }
}
