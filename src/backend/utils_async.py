# src/backend/utils_async.py
# I am not sure if i need it really, its kind of stops the active loop to work , i had to change all asyncio.run to run_async ? 
# dont know really why and how to fix the whole process to recieve all the logs 
import asyncio

try:
    import nest_asyncio  # pip install nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

def run_async(coro):
    """
    event loop (Jupyter/Streamlit).
    """
    try:
        loop = asyncio.get_running_loop() #
    except RuntimeError:
        # no active loop  can do default asyncio.run
        return asyncio.run(coro)
    else:
        # loop in the progress (Jupyter/Streamlit)  doing in it 
        return loop.run_until_complete(coro)
