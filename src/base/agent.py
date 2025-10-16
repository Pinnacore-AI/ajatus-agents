#!/usr/bin/env python3
# Ajatuskumppani — built in Finland, by the free minds of Pinnacore.

"""
AjatusAgents Base Agent

Base class for all autonomous agents in the Ajatuskumppani ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class BaseAgent(ABC):
    """
    Abstract base class for all Ajatus agents.
    
    All agents must implement:
    - execute(): Main execution logic
    - get_status(): Return current status
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_running = False
        self.last_run = None
        self.run_count = 0
        self.errors: List[str] = []
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Returns:
            Dictionary containing execution results
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "name": self.name,
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": len(self.errors),
            "config": self.config
        }
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the agent and handle errors.
        
        Returns:
            Execution results
        """
        self.is_running = True
        self.run_count += 1
        
        try:
            result = await self.execute()
            self.last_run = datetime.now()
            return result
            
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        finally:
            self.is_running = False
    
    def reset(self):
        """Reset the agent state"""
        self.is_running = False
        self.last_run = None
        self.run_count = 0
        self.errors = []


class TaskAgent(BaseAgent):
    """
    Agent for managing tasks and to-do lists.
    """
    
    async def execute(self) -> Dict[str, Any]:
        """Execute task management logic"""
        print(f"🔄 {self.name}: Checking tasks...")
        
        # TODO: Implement actual task management
        return {
            "tasks_checked": 10,
            "tasks_completed": 3,
            "tasks_pending": 7
        }


class NewsAgent(BaseAgent):
    """
    Agent for monitoring news feeds.
    """
    
    async def execute(self) -> Dict[str, Any]:
        """Execute news monitoring logic"""
        print(f"📰 {self.name}: Fetching news...")
        
        # TODO: Implement actual news fetching
        return {
            "articles_fetched": 25,
            "relevant_articles": 5
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create agents
        task_agent = TaskAgent("TaskAgent-1")
        news_agent = NewsAgent("NewsAgent-1")
        
        # Run agents
        task_result = await task_agent.run()
        news_result = await news_agent.run()
        
        # Print results
        print("\nTask Agent Result:", task_result)
        print("News Agent Result:", news_result)
        
        # Print status
        print("\nTask Agent Status:", task_agent.get_status())
        print("News Agent Status:", news_agent.get_status())
    
    asyncio.run(main())

