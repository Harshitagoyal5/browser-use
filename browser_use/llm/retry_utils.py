"""
Utility functions for LLM retry logic with timeout handling.

This module provides retry mechanisms for LLM calls that can be used across
different parts of the browser-use system.
"""

import asyncio
import logging
import time
from typing import Any

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage

logger = logging.getLogger(__name__)

async def ainvoke_with_retry_timeout(
    llm_instance: BaseChatModel, 
    messages: list[BaseMessage], 
    output_format: Any, 
    request_interval: int = 10, 
    max_retries: int = 3,
    logger_instance: logging.Logger | None = None
):
    """
    Send LLM requests with retry logic and timeout handling.
    
    Args:
        llm_instance: The LLM instance to use for the request
        messages: List of messages to send to the LLM
        output_format: The expected output format for the response
        request_interval: Time interval between retry attempts in seconds
        max_retries: Maximum number of retry attempts
        logger_instance: Optional logger instance to use for logging
        
    Returns:
        The first successful response from the LLM
        
    Raises:
        Exception: If all retry attempts fail
    """
    if logger_instance is None:
        logger_instance = logger
        
    start_time = time.time()
    logger_instance.debug(f"🚀 Starting LLM retry process (max {max_retries} attempts, {request_interval}s intervals)")
    tasks = []

    try:
        # Send requests with specified intervals
        for attempt in range(max_retries):
            # Start new request
            logger_instance.info(f"🔄 Starting LLM request attempt {attempt + 1}/{max_retries}")
    
            # Use the ainvoke method with output_format
            # Note: This will call the token-tracked version if the LLM is registered
            task = asyncio.create_task(llm_instance.ainvoke(messages, output_format=output_format))
            tasks.append(task)

            # Wait for any task to complete or timeout (except last attempt)
            if attempt < max_retries - 1:
                logger_instance.debug(f"⏳ Waiting up to {request_interval}s for any request to complete...")
                done, pending = await asyncio.wait(
                    tasks, 
                    timeout=request_interval, 
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Check if any task completed successfully
                for completed_task in done:
                    try:
                        result = await completed_task
                        # Success! Cancel remaining and return
                        elapsed_time = time.time() - start_time
                        logger_instance.info(f"✅ LLM request completed successfully in {elapsed_time:.2f}s! Cancelling {len([t for t in tasks if not t.done()])} remaining tasks")
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        return result
                    except Exception as e:
                        # Task failed, remove from list and continue
                        logger_instance.warning(f"❌ LLM request attempt failed: {str(e)}")
                        tasks.remove(completed_task)

                if pending:
                    logger_instance.debug(f"⏰ {request_interval}s timeout reached, {len(pending)} requests still running")

        # All requests sent, wait for any remaining to complete
        if tasks:
            logger_instance.info(f"🕐 All {max_retries} requests sent, waiting for any of {len(tasks)} remaining to complete...")
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    try:
                        result = await completed_task
       
                        # Success! Cancel remaining and return
                        elapsed_time = time.time() - start_time
                        logger_instance.info(f"✅ LLM request completed successfully in {elapsed_time:.2f}s! Cancelling {len([t for t in tasks if not t.done()])} remaining tasks")
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        return result
                    except Exception as e:
          
                        # Task failed, remove and continue
                        logger_instance.warning(f"❌ LLM request attempt failed: {str(e)}")
                        tasks.remove(completed_task)
       
        # All tasks failed
        elapsed_time = time.time() - start_time
        logger_instance.error(f"💥 All {max_retries} retry attempts failed after {elapsed_time:.2f}s")
        raise Exception("All retry attempts failed")

    except Exception as e:
        # Cancel all remaining tasks
        remaining_count = len([t for t in tasks if not t.done()])

        if remaining_count > 0:
            logger_instance.debug(f"🛑 Cancelling {remaining_count} remaining tasks due to error")

        for task in tasks:
            if not task.done():
                task.cancel()
        elapsed_time = time.time() - start_time
        logger_instance.error(f"❌ LLM retry process failed after {elapsed_time:.2f}s: {str(e)}")
     
        raise e
