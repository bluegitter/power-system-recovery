import ilog.concert.*;
import ilog.cplex.*;

public class PowerSystemRecovery {
    public static void main(String[] args) {
        try {
            IloCplex cplex = new IloCplex();

            // 数据初始化
            int Nbus = 14;
            int Ngen = 3;
            int Nbra = 20;
            int T = 120;
            int dt = 5;
            int Nt = T / dt;

            double[][] bus = {
                {1, 0, 0},
                {2, 21.7, 12.7},
                {3, 94.2, 19},
                {4, 47.8, -3.9},
                {5, 7.6, 1.6},
                {6, 11.2, 7.5},
                {7, 0, 0},
                {8, 0, 0},
                {9, 29.5, 16.6},
                {10, 3.5, 5.8},
                {11, 3.5, 1.8},
                {12, 6.1, 1.6},
                {13, 13.5, 5.8},
                {14, 14.9, 5}
            };

            double[][] gen = {
                {1, 332.4, 0, 1.11, 0, 0, 0, 900},
                {2, 140, 5, 1.37, 6, 11, 34.25, 900},
                {3, 100, 4, 1.27, 10, 15, 31.75, 900}
            };

            double[] genwi6 = {0.27, 0.27, 0.27, 0.38, 0.38, 0.38, 0.32, 0.32, 0.32, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.41, 0.41, 0.41};
            double[] genpv8 = {0.96, 0.96, 0.96, 1.85, 1.85, 1.85, 2.44, 2.44, 2.44, 3.41, 3.41, 3.41, 3.74, 3.74, 3.74, 4.18, 4.18, 4.18, 4.11, 4.11, 4.11, 5.40, 5.40, 5.40};

            int[][] bra = {
                {1, 1, 2},
                {2, 1, 5},
                {3, 2, 3},
                {4, 2, 4},
                {5, 2, 5},
                {6, 3, 4},
                {7, 4, 5},
                {8, 4, 7},
                {9, 4, 9},
                {10, 5, 6},
                {11, 6, 11},
                {12, 6, 12},
                {13, 6, 13},
                {14, 7, 8},
                {15, 7, 9},
                {16, 9, 10},
                {17, 9, 14},
                {18, 10, 11},
                {19, 12, 13},
                {20, 13, 14}
            };

            double M = 9999;

            // 定义优化变量
            IloNumVar[] P = new IloNumVar[Nt * Ngen];
            for (int i = 0; i < Nt * Ngen; i++) {
                P[i] = cplex.numVar(-Double.MAX_VALUE, Double.MAX_VALUE);
            }

            IloIntVar[][] a = new IloIntVar[Nbus][Nt];
            IloIntVar[][] b = new IloIntVar[Nbra][Nt];
            IloIntVar[][] k = new IloIntVar[Ngen][Nt];
            IloIntVar[][] x = new IloIntVar[Ngen][Nt];
            IloIntVar[][] y = new IloIntVar[Ngen][Nt];
            IloNumVar[][] PL = new IloNumVar[Nbus][Nt];
            IloIntVar[][] kk = new IloIntVar[2][Nt];

            for (int i = 0; i < Nbus; i++) {
                for (int t = 0; t < Nt; t++) {
                    a[i][t] = cplex.boolVar();
                    PL[i][t] = cplex.numVar(0, Double.MAX_VALUE);
                }
            }

            for (int i = 0; i < Nbra; i++) {
                for (int t = 0; t < Nt; t++) {
                    b[i][t] = cplex.boolVar();
                }
            }

            for (int i = 0; i < Ngen; i++) {
                for (int t = 0; t < Nt; t++) {
                    k[i][t] = cplex.boolVar();
                    x[i][t] = cplex.boolVar();
                    y[i][t] = cplex.boolVar();
                }
            }

            for (int i = 0; i < 2; i++) {
                for (int t = 0; t < Nt; t++) {
                    kk[i][t] = cplex.boolVar();
                }
            }

            // 定义目标函数
            IloLinearNumExpr objective = cplex.linearNumExpr();
            for (int i = 0; i < Nt * Ngen; i++) {
                objective.addTerm(1, P[i]);
            }
            for (int t = 0; t < Nt; t++) {
                objective.addTerm(genwi6[t] * dt, kk[0][t]);
                objective.addTerm(genpv8[t] * dt, kk[1][t]);
            }
            cplex.addMaximize(objective);

            // 添加约束条件
            // 连接性和次序约束
            for (int tt = 0; tt < Nt - 1; tt++) {
                int g = 0;
                int gg = 0;
                for (int i = 0; i < Nbus; i++) {
                    if (i == 0 || i == 1 || i == 2) {
                        g++;
                        cplex.addLe(k[g - 1][tt], a[i][tt]);
                    } else if (i == 5 || i == 7) {
                        gg++;
                        cplex.addLe(kk[gg - 1][tt], a[i][tt]);
                    }
                    IloLinearIntExpr sumExpr = cplex.linearIntExpr();
                    for (int l = 0; l < Nbra; l++) {
                        sumExpr.addTerm((int) (IM[l][i]), b[l][tt + 1]);
                    }
                    cplex.addLe(a[i][tt + 1], sumExpr);
                }
                for (int l = 0; l < Nbra; l++) {
                    cplex.addLe(b[l][tt + 1], cplex.sum(a[bra[l][1] - 1][tt], a[bra[l][2] - 1][tt]));
                }
            }

            // 其他约束
            for (int tt = 0; tt < Nt - 1; tt++) {
                for (int i = 0; i < Nbus; i++) {
                    cplex.addLe(a[i][tt], a[i][tt + 1]);
                }
                for (int g = 0; g < Ngen; g++) {
                    cplex.addLe(k[g][tt], k[g][tt + 1]);
                    cplex.addLe(x[g][tt], x[g][tt + 1]);
                    cplex.addLe(y[g][tt], y[g][tt + 1]);
                }
                for (int gg = 0; gg < 2; gg++) {
                    cplex.addLe(kk[gg][tt], kk[gg][tt + 1]);
                }
                for (int l = 0; l < Nbra; l++) {
                    cplex.addLe(b[l][tt], b[l][tt + 1]);
                }
            }

            // 初始化约束
            for (int i = 0; i < Ngen; i++) {
                cplex.addEq(k[i][0], (i == 0 ? 1 : 0));
            }
            for (int i = 1; i < Nbus; i++) {
                cplex.addEq(a[i][0], 0);
            }
            for (int l = 0; l < Nbra; l++) {
                cplex.addEq(b[l][0], 0);
            }
            for (int gg = 0; gg < 2; gg++) {
                cplex.addEq(kk[gg][0], 0);
            }

            // 连接性次序约束中的约束
            for (int tt = 0; tt < Nt - 1; tt++) {
                IloLinearIntExpr scha = cplex.linearIntExpr();
                for (int l = 0; l < Nbra; l++) {
                    scha.addTerm(1, b[l][tt + 1]);
                    scha.addTerm(-1, b[l][tt]);
                }
                cplex.addLe(scha, 1);
            }

            // 生成机组热启动时限约束
            for (int i = 0; i < Ngen; i++) {
                for (int tt = 0; tt < Nt; tt++) {
                    IloLinearIntExpr sumK = cplex.linearIntExpr();
                    for (int t = 0; t <= tt; t++) {
                        sumK.addTerm(1, k[i][t]);
                    }
                    cplex.addLe(sumK, cplex.sum(gen[i][4], y[i][tt]));
                    cplex.addGe(sumK, cplex.sum(gen[i][4], cplex.prod(-1, y[i][tt]), 1));
                    cplex.addLe(sumK, cplex.sum(gen[i][5], x[i][tt]));
                    cplex.addGe(sumK, cplex.sum(gen[i][5], cplex.prod(-1, x[i][tt]), 1));
                }
            }

            // 生成机组功率和爬坡率约束
            for (int i = 0; i < Ngen; i++) {
                for (int tt = 0; tt < Nt; tt++) {
                    cplex.addLe(P[i + Ngen * (tt - 1)], cplex.prod(M, k[i][tt]));
                    cplex.addGe(P[i + Ngen * (tt - 1)], cplex.prod(-M, k[i][tt]));
                    cplex.addLe(P[i + Ngen * (tt - 1)], cplex.sum(gen[i][3] * cplex.sum(y[i][tt], 1), cplex.prod(-gen[i][2], k[i][tt]), x[i][tt]));
                    cplex.addGe(P[i + Ngen * (tt - 1)], cplex.sum(gen[i][3] * cplex.sum(y[i][tt], 1), cplex.prod(-gen[i][2], k[i][tt]), cplex.prod(-x[i][tt], M)));
                    cplex.addLe(P[i + Ngen * (tt - 1)], cplex.sum(gen[i][6] * cplex.sum(1 - x[i][tt]), cplex.prod(-gen[i][2], k[i][tt]), M));
                    cplex.addGe(P[i + Ngen * (tt - 1)], cplex.sum(gen[i][1], cplex.prod(-gen[i][2], k[i][tt]), cplex.prod(-(1 - x[i][tt]), M)));
                    cplex.addLe(cplex.sum(tt, cplex.prod(-1, k[i][tt])), gen[i][7] - 1);
                }
            }

            // 生成机组爬坡率约束
            for (int i = 0; i < Ngen; i++) {
                for (int tt = 0; tt < Nt - 1; tt++) {
                    cplex.addLe(cplex.sum(P[i + Ngen * tt], cplex.prod(-1, P[i + Ngen * (tt - 1)])), gen[i][3] * dt);
                    cplex.addGe(cplex.sum(P[i + Ngen * tt], cplex.prod(-1, P[i + Ngen * (tt - 1)])), -gen[i][3] * dt);
                }
            }

            // 系统功率平衡约束
            for (int tt = 0; tt < Nt; tt++) {
                for (int i = 0; i < Nbus; i++) {
                    if (i == 10) {
                        cplex.addLe(PL[i][tt], cplex.prod(bus[i][1], a[i][tt]));
                        cplex.addGe(PL[i][tt], -3.5 * a[i][tt]);
                    } else {
                        cplex.addLe(PL[i][tt], cplex.prod(bus[i][1], a[i][tt]));
                        cplex.addGe(PL[i][tt], 0);
                    }
                }
                IloLinearNumExpr balance = cplex.linearNumExpr();
                for (int i = 0; i < Ngen; i++) {
                    balance.addTerm(1, P[i + Ngen * (tt - 1)]);
                }
                balance.addTerm(genwi6[tt], kk[0][tt]);
                balance.addTerm(genpv8[tt], kk[1][tt]);
                for (int i = 0; i < Nbus; i++) {
                    balance.addTerm(-1, PL[i][tt]);
                }
                cplex.addEq(balance, 0);
            }

            // 已恢复的负荷不再切除
            for (int tt = 0; tt < Nt - 1; tt++) {
                for (int i = 0; i < Nbus; i++) {
                    if (i != 10) {
                        cplex.addLe(PL[i][tt], PL[i][tt + 1]);
                    }
                }
            }

            // 求解模型
            if (cplex.solve()) {
                System.out.println("Total load restored: " + cplex.getObjValue());
                for (int i = 0; i < Nt * Ngen; i++) {
                    System.out.println("P[" + i + "] = " + cplex.getValue(P[i]));
                }
            } else {
                System.out.println("Solution not found");
            }
        } catch (IloException e) {
            e.printStackTrace();
        }
    }
}
