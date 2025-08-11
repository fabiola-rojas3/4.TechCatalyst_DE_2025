# Cron Example

Create a Python script as shown 

```python
#!/usr/bin/env python3

from datetime import datetime

def main():
    # Print a message with the current timestamp
    print(f"Hello from cron job! Current time: {datetime.now()}")

if __name__ == "__main__":
    main()

```

You can save it as `cron.py`

Create a shell script and save it as `run_every_5s.sh`

```shell
#!/bin/bash
for i in {1..12}
do
    /home/codespace/.python/current/bin/python /workspaces/TechCatalyst_DE_2025/code.py
    sleep 5
done
```

Let’s make sure it is executable

``` 
chmod +x ./run_every_5s.sh
```

Test it

```
./run_every_5s.sh
```

Modify your `crontab` entry to change to your workspace directory before running the script.

```
crontab -e
```

Just hit enter to select the best editor option (nano) and add the following. Just make sure it is pointing to your file if it is different 

```
* * * * * cd /workspaces/TechCatalyst_DE_2025 && /bin/bash ./run_every_5s.sh >> /workspaces/TechCatalyst_DE_2025/cron.log 2>&1
```

You can check cron status:

```
sudo service cron status
```

You can start the service 

```
sudo service cron start
```

You can stop the job by editing the file and commenting out the line you added. 

* Go to the line for your job and add a **`#`** symbol at the very beginning. This turns the line into a comment, and `cron` will ignore it.

Or, to stop all cron jobs you can run

```
crontab -r
```

> [!warning]
>
> **Warning:** This will remove **all** scheduled jobs, not just the one you created