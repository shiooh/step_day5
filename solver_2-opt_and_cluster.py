#!/usr/bin/env python3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import print_tour, read_input

import math
import numpy as np

class TSPSolver:
    def __init__(self, cities, x_split_num, y_split_num):
        self.cities = cities
        self.route_graph = [[] for _ in range(len(self.cities))]  # 道順を表すグラフ. ノードi に隣接するノードj を route_graph[i] に格納する
        self.clusters = []     # 各クラスターに含まれる id の配列. 
        self.x_split_num = x_split_num
        self.y_split_num = y_split_num
    
    ## city[id] が range に含まれるか計算する
    def is_id_in_range(self, id, range):
        if range['x_min'] >= self.cities[id][0]:
            return False
        if range['x_max'] < self.cities[id][0]:
            return False
        if range['y_min'] >= self.cities[id][1]:
            return False
        if range['y_max'] < self.cities[id][1]:
            return False
        return True

    ## city の座標に応じてクラスタリングし, self.clusters に city の id をクラスターごとに格納する
    def make_clusters(self):

        ## 各区間の範囲を計算し cluster_ranges に保存
        cluster_ranges = []

        # ± 1 の余裕をもった cities 全体の範囲
        total_x_min = min([self.cities[i][0]for i in range(len(self.cities))])-1
        total_x_max = max([self.cities[i][0]for i in range(len(self.cities))])+1
        total_y_min = min([self.cities[i][1]for i in range(len(self.cities))])-1
        total_y_max = max([self.cities[i][1]for i in range(len(self.cities))])+1

        # 区間(i, j)の範囲を計算. min は境界をふくみ, max は境界をふくまない.
        for i in range(self.x_split_num):
            for j in range(self.y_split_num):
                cluster_ranges.append({
                    'x_min': total_x_min + (total_x_max - total_x_min) / self.x_split_num * i,
                    'x_max': total_x_min + (total_x_max - total_x_min) / self.x_split_num * (i+1),
                    'y_min': total_y_min + (total_y_max - total_y_min) / self.y_split_num * j,
                    'y_max': total_y_min + (total_y_max - total_y_min) / self.y_split_num * (j+1),
                })
        
        ## 各区間に含まれる id の配列を raw_clusters に保存
        raw_clusters = [[] for _ in range(len(cluster_ranges))]
        for id in range(len(self.cities)):
            for i, cluster_range in enumerate(cluster_ranges):
                if self.is_id_in_range(id, cluster_range):
                    raw_clusters[i].append(id)
                    break
        
        ## 要素数が 4 未満の区間を隣の区間と統合して, 真のclustersを保存
        for cluster in raw_clusters:
            if len(self.clusters) == 0:
                self.clusters.append(cluster)
            elif len(cluster) < 4:
                self.clusters[-1].extend(cluster)
            else:
                self.clusters.append(cluster)


    ## city 同士の距離を計算する. 
    def calc_dist_of_cities(self, city_id_1, city_id_2):
        (x1, y1) = self.cities[city_id_1]
        (x2, y2) = self.cities[city_id_2]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    ## city 同士をグラフ上でつなげて, 巡回路長を更新する.
    def connect_cities(self, city_id_1, city_id_2):
        self.route_graph[city_id_1].append(city_id_2)
        self.route_graph[city_id_2].append(city_id_1)

    ## 指定のクラスター内の点の最短部分巡回路を貪欲法で求め, self.route_graph に道順を保存.
    def tsp_greedy(self, cluster_id):
        INF = 10**9     # 初期値: 無限遠にある点同士の距離

        ans = []
        visited = [False] * len(self.cities)

        cluster = self.clusters[cluster_id] # 指定のクラスター
        root_id = cluster[0]  # 指定のクラスター内の root_id から探索を始める
        cur_city_id = root_id
        visited[root_id] = True

        while True:
            # 未訪で cur_city_id に最も近い点 city_id_of_min_dist を探す
            min_dist = INF
            city_id_of_min_dist = -1
            for city_id in cluster:
                if not visited[city_id]:
                    this_dist = self.calc_dist_of_cities(cur_city_id, city_id)
                    if this_dist < min_dist:
                        min_dist = this_dist
                        city_id_of_min_dist = city_id
            
            # city が見つからなかったら探索終了
            if city_id_of_min_dist == -1:
                break
            # 見つかったらその点とのパスを route_graph に追加して次の探索へ
            else:
                self.connect_cities(cur_city_id, city_id_of_min_dist)
                cur_city_id = city_id_of_min_dist
                visited[cur_city_id] = True
        
        # 最後に root_id にもどるルートを追加
        self.connect_cities(cur_city_id, root_id)


    ## 有向パス（idのタプル）の次の有向パスを求める
    def next_path(self, path):
        (prev_id, cur_id) = path
        if self.route_graph[cur_id][0] != prev_id:
            next_id = self.route_graph[cur_id][0]
        else:
            next_id = self.route_graph[cur_id][1]
        return (cur_id, next_id)

    ## id から cur_neighbor までの有向辺を削除し new_neighbor までの有向辺を追加する
    def change_path(self, id, cur_neighbor, new_neighbor):
        if self.route_graph[id][0] == cur_neighbor:
            self.route_graph[id][0] = new_neighbor
        else:
            self.route_graph[id][1] = new_neighbor

    ## path1 と path2 が 交差していたら交換する. 返り値は, 交換した結果のパス2つ, または交換しなかったら元のパス2つに加え, 交換したかを示す bool 値.
    # path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 とする. id1⇔id4, id2⇔id3 とするとグラフが連結でなくなるので不可.
    def swap_crossing_paths(self, path1, path2):
        (id1, id2) = path1
        (id3, id4) = path2
        (x1, y1) = self.cities[id1]
        (x2, y2) = self.cities[id2]
        (x3, y3) = self.cities[id3]
        (x4, y4) = self.cities[id4]
        
        # 線分 id1,id2 と線分 id3,id4 の交点 P が 線分 id1,id2 を s:(1-s) で, 線分 id3,id4 を t:(1-t) で
        # 内分（s, tが負のときは外分）するとすると, x = (s, t)^T は Ax = b を満たす.
        A = np.array([
            [x1-x2, x4-x3],
            [y1-y2, y4-y3]
        ])
        b = np.array([x4-x2, y4-y2])
        x = np.linalg.solve(A, b)

        # 0 < s, t < 0 のとき, 線分上で交差しているのでパスを交換する.
        almost_0 = 0.0001
        if almost_0 < x[0] < 1-almost_0 and almost_0 < x[1] < 1-almost_0:
            self.change_path(id1, id2, id3)
            self.change_path(id2, id1, id4)
            self.change_path(id3, id4, id1)
            self.change_path(id4, id3, id2)
            return ((id1, id3), (id2, id4), True)
        else:
            return (path1, path2, False)

    ## 指定のクラスターの部分巡回路を 1 巡して, パス同士が交差していたら交換する.
    def apply_2opt_onece(self, cluster_id):
        if len(self.clusters[cluster_id]) < 4:
            return False
        changed_graph = False
        root_id = self.clusters[cluster_id][0]
        path1 = (root_id, self.route_graph[root_id][0])

        while True:
            path2 = self.next_path(path1)
            path2 = self.next_path(path2)
            if path2[0] == root_id:
                break
            while path2[0] != root_id:
                (path1, path2, changed_path) = self.swap_crossing_paths(path1, path2)
                changed_graph = changed_graph or changed_path
                path2 = self.next_path(path2)
            path1 = self.next_path(path1)
        return changed_graph

    ## 指定のクラスターの部分巡回路に 2-opt 法を適用する. パスを1巡して交差したパスを交換するだけでは再び別のパスが交差する可能性があるので, 複数回繰り返す.
    def apply_2opt(self, cluster_id):
        number_of_2opt = 10
        for i in range(number_of_2opt):
            changed_graph = self.apply_2opt_onece(cluster_id)
            if not changed_graph:
                break

    ## 指定のクラスター内の点の部分巡回路を貪欲法と2-opt 法で求める.
    def solve_cluster(self, cluster_id):
        self.tsp_greedy(cluster_id)
        self.apply_2opt(cluster_id)
    
    ## root_id からグラフに沿って id をめぐって, 通った id を配列に書き出す. グラフが連結でない場合は, root_id を含む連結部分のみ書き出される.
    def graph_to_city_id_tour(self, root_id=0):
        city_id_tour = []
        prev_id = root_id
        city_id_tour.append(prev_id)
        cur_id = self.route_graph[prev_id][0]

        while cur_id != root_id:
            city_id_tour.append(cur_id)
            (prev_id, cur_id) = self.next_path((prev_id, cur_id))
        
        return city_id_tour

    ## root_id からグラフに沿って id をめぐって, 通ったパス (start_id, end_id) を配列に書き出す. グラフが連結でない場合は, root_id を含む連結部分のみ書き出される.
    def graph_to_path_tour(self, cluster_id):
        path_tour = []
        root_id = self.clusters[cluster_id][0]
        cur_path = (root_id, self.route_graph[root_id][0])
        while True:
            path_tour.append(cur_path)
            cur_path = self.next_path(cur_path)
            if cur_path[0] == root_id:
                break
        return path_tour

    ## path の長さを計算する.
    def calc_path_dist(self, path):
        (x1, y1) = self.cities[path[0]]
        (x2, y2) = self.cities[path[1]]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    ## path1 と path2（2つは別のクラスターのパス）を繋ぎ替えた場合に, 全体の巡回路の長さがどれくらい増えるか計算する. 増えるとき正の数を返す.
    ## また, path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 と繋ぎ替えても id1⇔id4, id2⇔id3 と繋ぎ替えてもよいので,
    ## id1⇔id3, id2⇔id4 とした方が全体の巡回路が短くなる場合 "1to3_and_2to4" を, 他方の場合は "1to4_and_2to3" を返す.
    def calc_decrease_dist_and_better_swapping_way(self, path1, path2):
        (id1, id2) = path1
        (id3, id4) = path2
        cur_dist = self.calc_path_dist(path1) + self.calc_path_dist(path2)
        dist_of_1to3_and_2to4 = self.calc_path_dist((id1, id3)) + self.calc_path_dist((id2, id4))
        dist_of_1to4_and_2to3 = self.calc_path_dist((id1, id4)) + self.calc_path_dist((id2, id3))

        if dist_of_1to3_and_2to4 < dist_of_1to4_and_2to3:
            return (dist_of_1to3_and_2to4 - cur_dist, "1to3_and_2to4")
        else:
            return (dist_of_1to4_and_2to3 - cur_dist, "1to4_and_2to3")
        
    ## cluster[0] の部分巡回路に, 他のすべてのクラスターの部分巡回路を結合する
    def joint_clusters(self):
        INF = 10**9
        cluster_id_0 = 0
        disconned_cluster_ids = list(range(1, len(self.clusters)))  # まだ cluster[0] に結合していないクラスターの id

        # 全てが結合されるまで cluster_id_0 に他のクラスターを結合する.
        while len(disconned_cluster_ids) > 0:
            # cluster[0]　のパスと, 未結合のクラスターのパスの組み合わせのうち, それを繋ぎ替えたときに全体の巡回路の長さがが最も減る組み合わせを探す.
            # decrease_dist は, 経路が短くなる時 負の数とする.
            most_decrease_set = {'decrease_dist': INF}

            for path0 in self.graph_to_path_tour(cluster_id_0):
                for another_cluster_id in disconned_cluster_ids:
                    for path1 in self.graph_to_path_tour(another_cluster_id):
                        (decrease_dist, better_swapping_way) = self.calc_decrease_dist_and_better_swapping_way(path0, path1)
                        if decrease_dist < most_decrease_set['decrease_dist']:
                            most_decrease_set = {
                                'decrease_dist': decrease_dist,
                                'path0': path0,
                                'path1': path1,
                                'another_cluster_id': another_cluster_id,
                                'better_swapping_way': better_swapping_way
                            }
            
            # 見つかった組み合わせのパスを繋ぎ変える
            if most_decrease_set['better_swapping_way'] == "1to3_and_2to4":
                (id1, id2) = most_decrease_set['path0']
                (id3, id4) = most_decrease_set['path1']
                self.change_path(id1, id2, id3)
                self.change_path(id2, id1, id4)
                self.change_path(id3, id4, id1)
                self.change_path(id4, id3, id2)

            elif most_decrease_set['better_swapping_way'] == "1to4_and_2to3":
                (id1, id2) = most_decrease_set['path0']
                (id3, id4) = most_decrease_set['path1']
                self.change_path(id1, id2, id4)
                self.change_path(id2, id1, id3)
                self.change_path(id3, id4, id2)
                self.change_path(id4, id3, id1)

            else:
                return
            
            disconned_cluster_ids.remove(most_decrease_set['another_cluster_id'])
        self.clusters = [list(range(len(self.cities)))]

    ## 作成した部分巡回路をすべて可視化する
    def plot_graph(self, num=""):
        for cluster in self.clusters:
            root_id = cluster[0]
            city_id_tour = self.graph_to_city_id_tour(root_id)

            x = [self.cities[i][0] for i in city_id_tour]
            y = [-self.cities[i][1] for i in city_id_tour]
            x.append(self.cities[root_id][0])
            y.append(-self.cities[root_id][1])

            plt.plot(x,y,marker='.',markersize=5)
        plt.savefig(f'graph_{num}.png')
        plt.close()
        return

    def calc_tour_length(self):
        root_id = 0
        path_tour = self.graph_to_path_tour(root_id)
        tour_length = 0
        for path in path_tour:
            tour_length += self.calc_path_dist(path)
        return tour_length

    ## TSP を解く関数
    def solve(self):
        self.make_clusters()

        for i in range(len(self.clusters)):
            self.solve_cluster(i)
        self.plot_graph("cluster")
        self.joint_clusters()
        self.apply_2opt(0)
        self.plot_graph("joint")
        return (self.graph_to_city_id_tour(), self.calc_tour_length())

def solve(cities):
    INF = 10**9
    best_answer = ([], INF)
    for i in range(1,10):
        tsp_solver = TSPSolver(cities, i, i)
        answer = tsp_solver.solve()
        if answer[1] < best_answer[1]:
            best_answer = answer
    return best_answer[0]


if __name__ == '__main__':
    assert len(sys.argv) > 1
    tour = solve(read_input(sys.argv[1]))
    print_tour(tour)
