#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long
/*THIS COMMENT IS FOR GOOD LUCK*/
pair<int,int> decode(int x) {
	x--;
	return make_pair(x / 2000, x % 2000);
}
int encode(int x, int y) {
	return x * 2000 + y + 1;
}
int N;
int arr[1502][1502];
pair<int,int> child[1502][1502];
bool visited[1502][1502];
int goesTo[1502][1502];
//R = -1, D = -2
int ff(int onX, int onY, int val) {
	goesTo[onX][onY] = val;
	int tot = 0;
	if (onX - 1 >= 0 && arr[onX - 1][onY] == -2) {
		tot += ff(onX-1, onY, val);
	}
	if (onY-1 >= 0 && arr[onX][onY-1] == -1) {
		tot += ff(onX, onY-1, val);
	}
	return tot + 1;
}
ll calc() {
	ll tot = 0;
	for (int i = 0; i < N; i++) {
		tot += (ff(i, N, arr[i][N])-1) * arr[i][N];
	}
	for (int j = 0; j < N; j++) {
			tot += (ff(N, j, arr[N][j])-1) * arr[N][j];
	}
	return tot;
}
int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	//ifstream cin(".in");
	//ofstream cout(".out");
	cin >> N;
	for (int i = 0; i < N; i++) {
		string s; cin >> s;	int x; cin >> x;
		for (int j = 0; j < s.length(); j++) {
			if (s[j] == 'R') {
				arr[i][j] = -1;
			}
			else {
				arr[i][j] = -2;
			}
		}
		arr[i][N] = x;
	}
	for (int i = 0; i < N; i++) {
		cin >> arr[N][i];
	}
	int Q; cin >> Q;
	ll tot = calc();
	cout << tot << "\n";
	for (int i = 0; i < Q; i++) {
		int x, y; cin >> x >> y; x--; y--; 
		if (arr[x][y] == -2) {
			arr[x][y] = -1;
			tot -= (goesTo[x][y] - goesTo[x][y+1]) * ff(x, y, goesTo[x][y+1]);
		}
		else {
			arr[x][y] = -2;
			tot -= (goesTo[x][y] - goesTo[x+1][y]) * ff(x, y, goesTo[x+1][y]);
		}
		cout << tot << "\n";
	}

}
