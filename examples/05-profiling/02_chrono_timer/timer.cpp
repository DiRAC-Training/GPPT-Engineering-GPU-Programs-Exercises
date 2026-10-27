#include <chrono>
#include <cmath>
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
    return duration_cast<std::chrono::microseconds>(steady_clock::now() -
                                                    marks[mark])
        .count();
  }

  double timeBetween(int start, int end) {
    return duration_cast<std::chrono::microseconds>(marks[end] - marks[start])
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

float calculate(int N) {
  float result = 0.0f;
  for (int i = 0; i < N; i++) {
    result += std::sin(i * M_PI / N);
  }
  return result;
}

int main() {
  const int N = 1'000'000;

  {
    MarkTimer timer;

    int initTimerMark = timer.mark();
    float result = calculate(N);
    int finalTimerMark = timer.mark();

    printf("Time since beginning (ms): %f\n", timer.timeSince(initTimerMark));
    printf("Time in loop (ms): %f\n",
           timer.timeBetween(initTimerMark, finalTimerMark));
    printf("Result: %f\n", result);
  }

  {
    LapTimer<std::chrono::microseconds> timer;
    float result = calculate(N);
    printf("Time in loop (ms): %f\n", timer.lap());
    printf("Result: %f\n", result);
  }
}
