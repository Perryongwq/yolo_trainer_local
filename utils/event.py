class Event:
    """
    Simple event system for implementing the Observer pattern.
    Allows components to communicate without direct dependencies.
    """
    
    def __init__(self):
        """Initialize an empty list of subscribers"""
        self._subscribers = []
    
    def subscribe(self, callback):
        """
        Add a subscriber callback
        
        Args:
            callback: Function to call when the event is triggered
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
        return self  # Allow method chaining
    
    def unsubscribe(self, callback):
        """
        Remove a subscriber callback
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        return self  # Allow method chaining
    
    def trigger(self, *args, **kwargs):
        """
        Trigger the event, calling all subscribers.
        
        Exceptions in individual subscribers are caught and printed
        so that one failing callback does not break the remaining ones.
        
        Args:
            *args, **kwargs: Arguments to pass to subscribers
        """
        for subscriber in self._subscribers[:]:  # iterate over copy for safety
            try:
                subscriber(*args, **kwargs)
            except Exception as exc:
                print(f"[Event] subscriber {subscriber!r} raised {exc!r}")
    
    def __call__(self, *args, **kwargs):
        """
        Allow the event to be called directly as a function
        
        Args:
            *args, **kwargs: Arguments to pass to subscribers
        """
        self.trigger(*args, **kwargs)
