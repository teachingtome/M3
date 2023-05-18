#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long
/*THIS COMMENT IS FOR GOOD LUCK*/

vector<int> v;
int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	//ifstream cin(".in");
	//ofstream cout(".out");
	int T; cin >> T;
	for (int i = 0; i < T; i++) {
		int x; cin >> x;
		v.push_back(x);
	}
	for (int i = 0; i < T; i++) {
		cout << "R";
		v[i]-=2;
	}
	for (int j = T-1; j >= 0;) {
		if (v[j] == 0) {
			cout << "L";
			j--; continue;
		}
		int stop = j;
		int num = 1;
		for (int i = j-1; i >= 0; i--) {
			if (v[i] < v[j]) {
				stop = i + 1;
				break;
			}
			num++;
		}
		for (int i = 0; i < num; i++) {
			v[j - i] -= 2;
			cout << "L";
		}
		for (int i = 0; i < num; i++) {
			cout << "R";
		}
	}
}
