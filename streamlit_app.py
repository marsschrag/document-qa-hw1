import streamlit as st
st.title('HW Manager')
hw1 = st.Page('Homework/hw1.py', title = 'HW 1', icon = '⭐️')
hw2 = st.Page('Homework/hw2.py', title = 'HW 2', icon = '🌈')
hw3 = st.Page('Homework/hw3.py', title = 'HW 3', icon = '🍓')
hw4 = st.Page('Homework/hw4.py', title = 'HW 4', icon = '🌼')
pg = st.navigation([hw4, hw3, hw2, hw1])
st.set_page_config(page_title = 'HW Manager',
                   initial_sidebar_state = 'expanded')
pg.run()
