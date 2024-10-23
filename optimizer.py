#import necessary packages
import numpy as np
import sys
import math
import re
#import isoprene_rates as rate
from math import exp as EXP
from copy import deepcopy
import sympy as sym
#import networkx as nx
import matplotlib.pyplot as plt
#import graphviz
#import pygraphviz as pgv
#import to_precision
import time
import csv
import pandas as pd
import random

def anti_community_gd(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod):
  """### 5. Gradient Descent with Adam"""
  print()
  print("Beginning anti-community clustering gradient descent")

  def adam(h, coeffs, grads, learning_rate, epsilon=1e-8):

      curr_t = t[h]
      curr_m = m[h]
      curr_v = v[h]

      curr_t += 1
      new_coeffs = []


      for i, (coeff, g) in enumerate(zip(coeffs, grads)):
          curr_m[i] = beta1 * curr_m[i] + (1 - beta1) * g
          curr_v[i] = beta2 * curr_v[i] + (1 - beta2) * (g ** 2)

          m_hat = curr_m[i] / (1 - beta1 ** curr_t)
          v_hat = curr_v[i] / (1 - beta2 ** curr_t)

          coeff = coeff - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
          new_coeffs.append(coeff)

      t[h] = curr_t
      m[h] = curr_m
      v[h] = curr_v

      return new_coeffs

  t = []
  m = []
  v = []

  tick = time.time()
  new_prod_coeffs = copy_prod()
  history = []

  values_to_change = individual_coeffs  # value 1260 was found to have the most effect, and value 1102 was found to have the most effect out of any value that shared coeff yield changes with value 1260
  initial_score, initial_score_matrix = evaluate(new_prod_coeffs)

  initial_flat_score_matrix = []
  for i in range(len(initial_score_matrix)):
    for key in matrix_keys:
      initial_flat_score_matrix.append(initial_score_matrix[i][key])

  # Initial coefficients (you can start with any values)
  initial_coefficients = []
  for i in range(len(individual_coeff_map)):
      current_coeffs = []
      for n in range(len(individual_coeff_map[i])):
        j = map[values_to_change[i][n]][0]
        k = map[values_to_change[i][n]][1]
        current_coeffs.append(new_prod_coeffs[j][k])
      initial_coefficients.append(current_coeffs)


  # Set hyperparameters
  # iterations = 3
  # iterations = 1700
  history = []
  times = []
  gradient_sum_squares = []
  individual_coeff_score = []
  # learning_rate = 0.015
  learning_rates = []
  perturbation = []

  best_coeffs = []
  for i in range(len(initial_coefficients)):
    coeffs_list = []
    for j in range(len(initial_coefficients[i])):
      coeffs_list.append(initial_coefficients[i][j])
    best_coeffs.append(coeffs_list)

  best_score = initial_score

  # set initial perturbations, learning rates, and find initial individual coefficient scores
  for i in range(len(initial_coefficients)):
    gradient_sum_squares.append([])
    perturbation.append([])
    learning_rates.append([])
    individual_coeff_score.append([])
    t.append(0)
    m.append([])
    v.append([])
    for j in range(len(initial_coefficients[i])):
      m[i].append(0)
      v[i].append(0)
      learning_rates[i].append(learning_rate)
      # perturbation[i].append(initial_coefficients[i][j]*0.00000005)
      perturbation[i].append(initial_coefficients[i][j]*0.000005)
      gradient_sum_squares[i].append(0)
      score = 0
      for k in individual_coeff_map[i][j]:
        score += abs(initial_flat_score_matrix[k])
      individual_coeff_score[i].append(score)

  for x in range(iterations):
      # for h in range(5):
      # for h in [0]:
      for h in range(len(individual_coeffs)):
      # for h in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        values_to_change = individual_coeffs[h]
        gradient = np.zeros_like(initial_coefficients[h])

        # Update prod coeffs list
        for i in range(len(initial_coefficients[h])):
            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] + perturbation[h][i]

        # calculate costs with plus perturbations
        avg_score, score_matrix = evaluate(new_prod_coeffs)
        flat_score_matrix = []
        for i in range(len(score_matrix)):
          for key in matrix_keys:
            flat_score_matrix.append(score_matrix[i][key])

        plus_scores = []
        for i in range(len(initial_coefficients[h])):
          matrix_values = individual_coeff_map[h][i]
          total_score = 0
          for j in matrix_values:
            total_score += abs(flat_score_matrix[j])
            # total_score += flat_score_matrix[j]**2

          total_score = (total_score/len(matrix_values))
          plus_scores.append(total_score)


        plus_costs = np.multiply(plus_scores, plus_scores) # square costs

        for i in range(len(initial_coefficients[h])):
            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] - perturbation[h][i]

        # calculate costs with minus perturbations
        avg_score, score_matrix = evaluate(new_prod_coeffs)
        flat_score_matrix = []
        for i in range(len(score_matrix)):
          for key in matrix_keys:
            flat_score_matrix.append(score_matrix[i][key])

        minus_scores = []
        for i in range(len(initial_coefficients[h])):
          matrix_values = individual_coeff_map[h][i]
          total_score = 0
          for j in matrix_values:
            total_score += abs(flat_score_matrix[j])

          total_score = (total_score/len(matrix_values))
          minus_scores.append(total_score)


        minus_costs = np.multiply(minus_scores, minus_scores) # square costs

        gradients = []
        for i in range(len(plus_costs)):
          gradient = (plus_costs[i]-minus_costs[i])/perturbation[h][i]
          gradients.append(gradient)

        new_coeffs = adam(h, initial_coefficients[h], gradients, learning_rate=learning_rate)

        # Update coefficients and adapt perturbations and learning rates
        for i in range(len(initial_coefficients[h])):
          score = 0
          for k in individual_coeff_map[h][i]:
            score += abs(flat_score_matrix[k])

          individual_coeff_score[h][i] = score
          initial_coefficients[h][i] = new_coeffs[i]

          if initial_coefficients[h][i] < 0:
            initial_coefficients[h][i] = 0

      for h in range(len(initial_coefficients)):
        for i in range(len(initial_coefficients[h])):
            new_prod_coeffs[map[individual_coeffs[h][i]][0]][map[individual_coeffs[h][i]][1]] = initial_coefficients[h][i]

      avg_score, score_matrix = evaluate(new_prod_coeffs)

      history.append(avg_score)
      times.append(time.time()-tick)

      print((x+1)*100/iterations, "percent completed: The average score has been optimized by", ((initial_score - avg_score) / initial_score) * 100, "percent")

      if x>30:
        if abs((((initial_score - history[x-30]) / initial_score) * 100) - (((initial_score - avg_score) / initial_score) * 100)) < 0.5:
          all_bad = True
          for i in range(1, 15):
            if abs((((initial_score - history[x-i]) / initial_score) * 100) - (((initial_score - avg_score) / initial_score) * 100)) > 0.2:
              all_bad = False
          if all_bad:
            break

      if avg_score < best_score:
        best_score = avg_score
        best_coeffs = []
        for i in range(len(initial_coefficients)):
          coeffs_list = []
          for j in range(len(initial_coefficients[i])):
            coeffs_list.append(initial_coefficients[i][j])
          best_coeffs.append(coeffs_list)

  for i in range(len(best_coeffs)):
    for j in range(len(best_coeffs[i])):
      new_prod_coeffs[map[individual_coeffs[i][j]][0]][map[individual_coeffs[i][j]][1]] = best_coeffs[i][j]


  final_score, final_score_matrix = evaluate(new_prod_coeffs)
  tock = time.time()

  print(final_score_matrix)
  print("\nOptimized Coefficients:", best_coeffs)
  print("Final Score:", final_score)
  print("Optimized by", ((initial_score - final_score) / initial_score) * 100, "percent")
  print("Times array", times)
  print("Scores array", history)
  print("\nThe gradient descent", tock-tick, "seconds")

  plt.plot(times, history)
  return best_coeffs

def undirected_graph_sort(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod):
  print()
  print("Beginning undirected graph sort gradient descent")
  """### 6. Undirected Graph Sort"""

  # MAKE TOPOLOGICAL ORIENTATION OF GRAPH (reverse of yield_change_graph)
  topo_graph = []
  for j in range(len(yield_change_graph[0])):
    current_graph = []
    for row in yield_change_graph:
      current_graph.append(row[j])
    topo_graph.append(current_graph)

  # SORT GRAPH
  topo_sort_graph = []
  topo_coeffs2 = []
  topo_effected_yields = []
  rows_map = []
  topo_graph_dict = {}

  for i in range(len(topo_graph)):
    rows_map.append(i)
    topo_graph_dict[i] = topo_graph[i]


  rows_delete = []
  dict_keys = list(topo_graph_dict.keys())
  for i in dict_keys:
    counter = 0
    for j in range(len(topo_graph[i])):
      counter += topo_graph[i][j]
    if counter == 0:
      rows_delete.append(i)

  for i in rows_delete:
    del topo_graph_dict[i]


  # iterate sorting until there are no rows left
  while len(topo_graph_dict) > 0:
    counter_obj = {}
    rows_delete = []
    least = 1000
    greatest = 0
    dict_keys = list(topo_graph_dict.keys())

    # check how many connections all rows have
    for i in dict_keys:
      counter = 0
      for j in range(len(topo_graph_dict[i])):
        counter += topo_graph[i][j]
      # check if this row has the least amount of connections
      if counter < least and counter > 0:
        least = counter
      # find greatest number of connections just for informational purposes
      if counter > greatest:
        greatest = counter
      # update counting object with the number of connection this row has
      if counter in counter_obj:
        new_list = counter_obj[counter]
        counter_obj[counter].append(i)
      else:
        counter_obj[counter] = [i]


    # get all rows with the least amount of connections, and get all effecting coefficients in each of those rows
    least_rows = counter_obj[least]
    curr_topo_coeffs = []
    curr_effected_yields = []
    curr_topo_coeffs22 = []
    for i in least_rows:
      curr_topo_coeffs2 = []
      for j in range(len(topo_graph_dict[i])):
        if topo_graph_dict[i][j] == 1 and j not in curr_topo_coeffs:
          curr_topo_coeffs.append(j)
          curr_topo_coeffs2.append(j)
      curr_effected_yields.append(i)
      curr_topo_coeffs22.append(curr_topo_coeffs2)

    topo_sort_graph.append(curr_topo_coeffs)
    topo_effected_yields.append(curr_effected_yields)
    topo_coeffs2.append(curr_topo_coeffs22)

    # remove all connections with the current coefficients that we just got
    dict_keys = list(topo_graph_dict.keys())
    for i in dict_keys:
      for j in curr_topo_coeffs:
        topo_graph[i][j] = 0

    for i in dict_keys:
      counter = 0
      for j in range(len(topo_graph[i])):
        counter += topo_graph[i][j]
      if counter == 0:
        rows_delete.append(i)

    for i in rows_delete:
      del topo_graph_dict[i]

  counter = 0
  for row in topo_sort_graph:
    for num in row:
      counter += 1

  topo_map = [row[:] for row in topo_sort_graph]
  for i in range(len(topo_map)):
    for j in range(len(topo_map[i])):
      coeff = topo_map[i][j]
      yield_changes = yield_change_graph[coeff]
      current_changes = []
      for k in range(len(yield_changes)):
        if yield_changes[k] == 1:
          current_changes.append(k)
      topo_map[i][j] = current_changes

  topo_map2 = [row[:] for row in [row2[:] for row2 in [row3[:] for row3 in topo_coeffs2]]]
  topo_map2 = []
  for i in range(len(topo_coeffs2)):
    curr_topo_map = []
    for j in range(len(topo_coeffs2[i])):
      curr_topo_map2 = []
      for k in range(len(topo_coeffs2[i][j])):
        coeff = topo_coeffs2[i][j][k]
        yield_changes = yield_change_graph[coeff]
        current_changes = []
        for l in range(len(yield_changes)):
          if yield_changes[l] == 1:
            current_changes.append(l)
        curr_topo_map2.append(current_changes)
      curr_topo_map.append(curr_topo_map2)
    topo_map2.append(curr_topo_map)

  tick = time.time()
  new_prod_coeffs = copy_prod()
  history = []

  values_to_change = topo_sort_graph  # value 1260 was found to have the most effect, and value 1102 was found to have the most effect out of any value that shared coeff yield changes with value 1260
  initial_score, initial_score_matrix = evaluate(new_prod_coeffs)

  initial_flat_score_matrix = []
  for i in range(len(initial_score_matrix)):
    for key in matrix_keys:
      initial_flat_score_matrix.append(initial_score_matrix[i][key])

  # Initial coefficients (you can start with any values)
  initial_coefficients = []
  for i in range(len(topo_map)):
      current_coeffs = []
      for n in range(len(topo_map[i])):
        j = map[values_to_change[i][n]][0]
        k = map[values_to_change[i][n]][1]
        current_coeffs.append(new_prod_coeffs[j][k])
      initial_coefficients.append(current_coeffs)

  print("Initial Coefficients", initial_coefficients)
  print("Initial Score", initial_score)

  # Set hyperparameters
  max_iterations = iterations
  best_score = initial_score
  best_coeffs = initial_coefficients
  # iterations = 150
  history = []
  times = []
  history_raw = []
  perturbation = []
  initial_learning_rate = learning_rate
  learning_rate = []
  gradient_sum_squares = []
  individual_coeff_score = []

  # set initial perturbations, learning rates, and find initial individual coefficient scores
  for i in range(len(initial_coefficients)):
    perturbation.append([])
    learning_rate.append([])
    gradient_sum_squares.append([])
    individual_coeff_score.append([])
    for j in range(len(initial_coefficients[i])):
      learning_rate[i].append(initial_learning_rate)
      # learning_rate[i].append(0.02)
      gradient_sum_squares[i].append(0)
      perturbation[i].append(initial_coefficients[i][j]*0.000005)
      score = 0
      for k in topo_map[i][j]:
        score += abs(initial_flat_score_matrix[k])
      individual_coeff_score[i].append(score)

  # for h in range(5):
  # for h in [0]:
  for h in range(len(topo_sort_graph)):
      past_scores = []
      for x in range(max_iterations):
        gradients = []
        # for h in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        values_to_change = topo_sort_graph[h]
        gradient = np.zeros_like(initial_coefficients[h])

        # create a copy of the list and then update each coefficient one at a time, checking each gradient
        list_copy = [row[:] for row in new_prod_coeffs]
        for i in range(len(initial_coefficients[h])):
            new_prod_coeffs = [row[:] for row in list_copy]
            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] + perturbation[h][i]

            # calculate costs with plus perturbations
            avg_score, score_matrix = evaluate(new_prod_coeffs)
            flat_score_matrix = []
            for j in range(len(score_matrix)):
              for key in matrix_keys:
                flat_score_matrix.append(score_matrix[j][key])

            matrix_values = topo_map[h][i]
            total_score = 0
            for j in matrix_values:
              total_score += abs(flat_score_matrix[j])

            plus_score = (total_score/len(matrix_values))


            plus_cost = plus_score*plus_score # square costs


            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] - perturbation[h][i]

            # calculate costs with minus perturbations
            avg_score, score_matrix = evaluate(new_prod_coeffs)
            flat_score_matrix = []
            for j in range(len(score_matrix)):
              for key in matrix_keys:
                flat_score_matrix.append(score_matrix[j][key])

            history.append(avg_score)

            matrix_values = topo_map[h][i]
            total_score = 0
            for j in matrix_values:
              total_score += abs(flat_score_matrix[j])

            minus_score = (total_score/len(matrix_values))

            minus_cost = minus_score*minus_score # square costs

            gradients.append((plus_cost-minus_cost)/perturbation[h][i])

        # Update coefficients and adapt perturbations and learning rates
        for i in range(len(initial_coefficients[h])):
          score = 0
          for k in topo_map[h][i]:
            score += abs(flat_score_matrix[k])

          gradient_sum_square = gradient_sum_squares[h][i] + gradients[i]**2
          gradient_sum_squares[h][i] = gradient_sum_square
          new_learning_rate = learning_rate[h][i]/np.sqrt(gradient_sum_square + 1e-8)


          if score <= individual_coeff_score[h][i]:
            # learning_rate[h][i] = learning_rate[h][i]*1.09
            learning_rate[h][i] = learning_rate[h][i]*1.05
            # perturbation[h][i] = perturbation[h][i]*1.06
          else:
            learning_rate[h][i] = learning_rate[h][i]*0.8
            perturbation[h][i] = perturbation[h][i]*0.8

          individual_coeff_score[h][i] = score

          gradient = (new_learning_rate * gradients[i])
          initial_coefficients[h][i] -= gradient

          if initial_coefficients[h][i] < 0:
            initial_coefficients[h][i] = 0

          new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i]

        times.append(time.time()-tick)
        history_raw.append(avg_score)

        print((h+x*0.001+1)*100/(len(topo_sort_graph)), "percent completed: The average score has been optimized by", ((initial_score - avg_score) / initial_score) * 100, "percent")
        past_scores.append(((initial_score - avg_score) / initial_score) * 100)
        if avg_score < best_score:
          best_score = avg_score
          best_coeffs = initial_coefficients
        if x>20:
          if past_scores[x-20] > (((initial_score - avg_score) / initial_score) * 100)-0.1:
            initial_coefficients = best_coeffs
            break

  final_score, final_score_matrix = evaluate(new_prod_coeffs)
  tock = time.time()

  print(final_score_matrix)
  print("\nOptimized Coefficients:", initial_coefficients)
  print("Final Score:", final_score)
  print("Optimized by", ((initial_score - final_score) / initial_score) * 100, "percent")
  print("Times array", times)
  print("Scores array", history_raw)
  print("\nThe gradient descent", tock-tick, "seconds")

  return initial_coefficients


def standard_gd(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod):
  print()
  print("Beginning standard gradient descent")
  """### 7. Standard GD (Avg Score)"""

  def adam(h, coeffs, grads, learning_rate, epsilon=1e-8,):

      curr_t = t[h]
      curr_m = m[h]
      curr_v = v[h]

      curr_t += 1
      new_coeffs = []


      for i, (coeff, g) in enumerate(zip(coeffs, grads)):
          curr_m[i] = beta1 * curr_m[i] + (1 - beta1) * g
          curr_v[i] = beta2 * curr_v[i] + (1 - beta2) * (g ** 2)

          m_hat = curr_m[i] / (1 - beta1 ** curr_t)
          v_hat = curr_v[i] / (1 - beta2 ** curr_t)

          coeff = coeff - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
          new_coeffs.append(coeff)

      t[h] = curr_t
      m[h] = curr_m
      v[h] = curr_v

      return new_coeffs

  import numpy as np

  t = []
  m = []
  v = []

  tick = time.time()
  new_prod_coeffs = copy_prod()
  history = []

  values_to_change = individual_coeffs  # value 1260 was found to have the most effect, and value 1102 was found to have the most effect out of any value that shared coeff yield changes with value 1260
  initial_score, initial_score_matrix = evaluate(new_prod_coeffs)

  initial_flat_score_matrix = []
  for i in range(len(initial_score_matrix)):
    for key in matrix_keys:
      initial_flat_score_matrix.append(score_matrix[i][key])

  # Initial coefficients (you can start with any values)
  initial_coefficients = []
  for i in range(len(individual_coeff_map)):
      current_coeffs = []
      for n in range(len(individual_coeff_map[i])):
        j = map[values_to_change[i][n]][0]
        k = map[values_to_change[i][n]][1]
        current_coeffs.append(new_prod_coeffs[j][k])
      initial_coefficients.append(current_coeffs)

  print("Initial Coefficients", initial_coefficients)
  print("Initial Score", initial_score)

  # Set hyperparameters
  # iterations = 1000
  # iterations = 2000
  times = []
  history = []
  gradient_sum_squares = []
  individual_coeff_score = []
  # learning_rate = 0.015
  best_score = initial_score
  best_coeffs = []
  for i in range(len(initial_coefficients)):
    coeffs_list = []
    for j in range(len(initial_coefficients[i])):
      coeffs_list.append(initial_coefficients[i][j])
    best_coeffs.append(coeffs_list)
  learning_rates = []
  perturbation = []

  # set initial perturbations, learning rates, and find initial individual coefficient scores
  for i in range(len(initial_coefficients)):
    gradient_sum_squares.append([])
    perturbation.append([])
    learning_rates.append([])
    individual_coeff_score.append([])
    t.append(0)
    m.append([])
    v.append([])
    for j in range(len(initial_coefficients[i])):
      m[i].append(0)
      v[i].append(0)
      learning_rates[i].append(learning_rate)
      # perturbation[i].append(initial_coefficients[i][j]*0.00000005)
      perturbation[i].append(initial_coefficients[i][j]*0.000005)
      gradient_sum_squares[i].append(0)
      score = 0
      for k in individual_coeff_map[i][j]:
        score += abs(initial_flat_score_matrix[k])
      individual_coeff_score[i].append(score)

  for x in range(iterations):
      for h in range(len(individual_coeffs)):
        values_to_change = individual_coeffs[h]
        gradients = []

        # Update prod coeffs list
        for i in range(len(initial_coefficients[h])):
            new_prod_coeffs = copy_prod()
            for j in range(len(initial_coefficients)):
              for k in range(len(initial_coefficients[j])):
                new_prod_coeffs[map[individual_coeffs[j][k]][0]][map[individual_coeffs[j][k]][1]] = initial_coefficients[j][k]

            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] + perturbation[h][i]

            # calculate costs with plus perturbations
            avg_score, score_matrix = evaluate(new_prod_coeffs)
            plus_cost = avg_score*avg_score

            new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = initial_coefficients[h][i] - perturbation[h][i]

            # calculate costs with minus perturbations
            avg_score, score_matrix = evaluate(new_prod_coeffs)
            minus_cost = avg_score*avg_score

            gradient = (plus_cost-minus_cost)/perturbation[h][i]
            gradients.append(gradient)

        new_coeffs = adam(h, initial_coefficients[h], gradients, learning_rate=learning_rate)

        # Update coefficients and adapt perturbations and learning rates
        for i in range(len(initial_coefficients[h])):
          initial_coefficients[h][i] = new_coeffs[i]

          if initial_coefficients[h][i] < 0:
            initial_coefficients[h][i] = 0

      history.append(avg_score)
      times.append(time.time()-tick)

      if avg_score < best_score:
        best_score = avg_score
        best_coeffs = []
        for i in range(len(initial_coefficients)):
          coeffs_list = []
          for j in range(len(initial_coefficients[i])):
            coeffs_list.append(initial_coefficients[i][j])
          best_coeffs.append(coeffs_list)

      if x>20:
        if (((initial_score - history[x-20]) / initial_score) * 100) > (((initial_score - avg_score) / initial_score) * 100)-0.05:
          all_bad = True
          for i in range(1, 10):
            if (((initial_score - history[x-i]) / initial_score) * 100) < (((initial_score - avg_score) / initial_score) * 100)-0.05:
              all_bad = False
          if all_bad:
            break

      print((x+1)*100/iterations, "percent completed: The average score has been optimized by", ((initial_score - avg_score) / initial_score) * 100, "percent")

  for i in range(len(best_coeffs)):
    for j in range(len(best_coeffs[i])):
      new_prod_coeffs[map[individual_coeffs[i][j]][0]][map[individual_coeffs[i][j]][1]] = best_coeffs[i][j]


  final_score, final_score_matrix = evaluate(new_prod_coeffs)
  tock = time.time()

  print(final_score_matrix)
  print("\nOptimized Coefficients:", best_coeffs)
  print("Final Score:", final_score)
  print("Optimized by", ((initial_score - final_score) / initial_score) * 100, "percent")
  print("Times array", times)
  print("Scores array", history)
  print("\nThe gradient descent", tock-tick, "seconds")

  return best_coeffs

def f0am_file(prod_list_n_r, mech_name, copy_prod, optimized_coeffs, individual_coeffs, map):
  """### Create f0am File

  Make prod coeff list with our coeffs
  """

  # set our coefficients
  # initial_coefficients = [[6.103286364346059, 7.029856289430873, 0, 0, 0, 4.84191943494451, 0.9243617644534453, 0, 6.215205162275304, 0, 3.2940156755667998], [4.9941901033123886, 1.6358405251378647, 0, 7.981480206991661, 0, 0, 0, 9.93422210054361, 0, 0, 0], [4.8678811645250555, 0.2794876746399177, 0, 0, 0.8945465832793635, 1.1496558833327433, 0, 2.377607297559912], [5.189446674226551, 0, 2.4016590162231948, 0.07360323417099703, 0.146161588770096], [1.0228326131146226, 2.3299096334191596, 0.016119283889303553], [0.6880988979545873, 2.225240560477585], [0, 0, 1.7631716793215342, 6.764312004269307, 0, 8.315105397537797, 0, 5.190408618767493], [0, 0, 0.025905877777998697, 0, 1.6489566374977533, 0, 8.628484580774584], [0, 0, 0, 0, 0.07356279822465539], [1.1893794138233957, 1.9214338911235849, 0, 0.008258281923444442, 0], [0, 0, 0], [0, 0], [0, 0], [5.329225998376063], [0], [0.04633996004069245, 0.26347402292873445, 0], [0.18440494523311868, 0, 0], [5.62931716990747], [0], [0.06757405700073058, 0, 0, 0], [0, 0.30179232276842777, 0], [0, 0.7556905385369188], [0, 1.0011438732608509], [0, 1.105630349804568], [0, 0.726947568829931, 0, 0], [0, 0, 0], [1.4109207327966249, 0], [1.873808830919395, 0.008044541177947397, 0, 0], [0.9343317109580763, 0, 0], [1.929896321066118, 0.07762237115414321, 0, 0], [0.12455366376232976, 0, 0.5969599258574305], [0.5701543103677081], [0.1963617159225069], [0, 0.23588822931317957, 0], [0, 2.0905066917126236]]
  new_prod_coeffs = copy_prod()
  for h in range(len(individual_coeffs)):
      values_to_change = individual_coeffs[h]
      gradient = np.zeros_like(optimized_coeffs[h])
      for i in range(len(optimized_coeffs[h])):
          new_prod_coeffs[map[values_to_change[i]][0]][map[values_to_change[i]][1]] = optimized_coeffs[h][i]

  # reactions is a list and each reaction is the
  class Mechanism:
      def __init__(self, species, reactions):
          self.species = species
          self.reactions = reactions

  class Reaction:
      def __init__(self, reactants, prod_dict, rate_law, eval_rate_law, rate, rate_string = '', multiplier = 1):
          self.reactants = reactants
          self.prod_dict = prod_dict
          self.rate_law = rate_law
          self.eval_rate_law = eval_rate_law
          self.rate = rate
          self.rate_string = rate_string
          self.multiplier = multiplier

  reactions = []
  for i in range(len(prod_list_n_r)):
    prod_dict = {prod_list_n_r[i][j]: new_prod_coeffs[i][j] for j in range(len(prod_list_n_r[i]))}
    rxn = Reaction(reac_list_n_r[i], prod_dict, rates_2[i], 1, 1, 1, 1)
    reactions.append(rxn)

  optimized_mech = Mechanism(species_list_names_r, reactions)

  #create_f0am_file(network,reaction_list,species_list,name)
  def create_f0am_file_no_rate_change(mech,name):
      spec_2_add = "SpeciesToAdd = {'ISOPN'; "
      count = 0
      for i in mech.species:
          count+=1
          spec_2_add = spec_2_add + "'" + i +"'"+ ';'
      spec_2_add = spec_2_add[:-1]
      spec_2_add = spec_2_add  + "};"

      eq_str = ''
      for i in range(len(mech.reactions)):
          r_string = ''
          for j in mech.reactions[i].reactants:
              r_string = r_string + str(j) + ' + '
          r_string = r_string[:-2] + '= '
          for j in mech.reactions[i].prod_dict:
              r_string = r_string + str(j) + ' + '
          r_string = r_string[:-3]
          reac_str = ''
          reac_str = reac_str + "\ni=i+1;\nRnames{i} = '" + r_string + "';\nk(:,i) = "+ str(mech.reactions[i].rate_law)+ '*'+str(mech.reactions[i].multiplier) + ';\n'
          counter = 1
          for j in mech.reactions[i].reactants:
              reac_str = reac_str + 'Gstr{i,'+str(counter)+"} = '"+str(j)+"'; "
              counter = counter + 1
          reac_str = reac_str +'\n'
          for k in mech.reactions[i].reactants:
              reac_str = reac_str + 'f'+ str(k) +'(i)'+'='+'f'+ str(k) +'(i)'+'-1' + '; '
          for k in mech.reactions[i].prod_dict:
              reac_str = reac_str + 'f'+ str(k)+'(i)'+'='+'f'+ str(k)+'(i)'+'+'+str(mech.reactions[i].prod_dict[k]) + '; '

          reac_str = reac_str +'\n'
          reac_str = reac_str +'\n'
          eq_str = eq_str + reac_str

      full = spec_2_add + '\n'+'RO2ToAdd = {};'+'\n'+'AddSpecies'+'\n'+ eq_str
      f0am_file = open("f0am_"+name+".m","w+")
      f0am_file.write(full)

  create_f0am_file_no_rate_change(optimized_mech, mech_name)

from setup import *

def AMORE_Optimization(iterations=400, learning_rate=0.015, input_conditions=1, method="anticommunity", individual_params=1, lower_limit=0, upper_limit=1000, beta1=0.9, beta2=0.9, mech_name="optimized_mechanism"):
  # setup.config = {'mech_file': mech_file, 'full_eqn': full_eqn, 'full_spc': full_spc, 'input_conditions': input_conditions, 'individual_params': 1}
  # initialize({'mech_file': mech_file, 'full_eqn': full_eqn, 'full_spc': full_spc, 'input_conditions': input_conditions, 'individual_params': 1})
  # lower_limit=0, upper_limit=1000, alpha=0.9, beta=0.9, learning_rate, 'method': method
  new_prod_coeffs = copy_prod()
  avg_score, score_matrix = evaluate(new_prod_coeffs)
  optimized_coeffs = [[]]
  valid_method = True

  if method == "anticommunity":
    optimized_coeffs = anti_community_gd(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod)
  elif method == "undirectedgraph":
    optimized_coeffs = undirected_graph_sort(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod)
  elif method == "standard":
    optimized_coeffs = standard_gd(iterations, learning_rate, beta1, beta2, individual_coeffs, evaluate, matrix_keys, individual_coeff_map, copy_prod)
  else:
    print("Please choose a valid optimization method: anticommunity, undirectedgraph, or standard")
    valid_method = False
  
  if valid_method:
    f0am_file(prod_list_n_r, mech_name, copy_prod, optimized_coeffs, individual_coeffs, map)