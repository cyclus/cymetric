.. _cym_tutorial:

Tutorial
========
This document will walk you through the basics of how to use cymetric on the 
command line, from Python, and even how to write your own metrics!

Cymetric operates by reading data from a |cyclus| database, computing metrics, 
and writing those metrics back to the database. This way, previously seen 
metrics are stored for later retrieval.  The dependencies between metrics are 
automatically computed and evaluated.  

Without further ado, let's dive in!

Command Line Usage
------------------
Cymetric ships with a command line utility, just called ``cymetric``. Since 
cymetric is wholly dependent on acting |cyclus| database, you must supply a
databse as an argument on the command line. We'll be using ``test.h5`` and 
``test.sqlite`` as our example database here.  

Table Listing: ``-l``
~~~~~~~~~~~~~~~~~~~~~
The first switch here, lowercase-L ``-l``, simply lists the tables in the database.
Note that this may list more tables in the database than are strictly part of the 
|cyclus| interface, because of the need for |cyclus| to store metadata. Don't
be alarmed. Listing the current tables is easy:

.. code-block:: bash

    $ cymetric test.h5 -l
    AgentEntry
    AgentStateAgent
    AgentStateInventories
    AgentState_agents_NullRegionInfo
    ...
    BlobKeys
    BlobVals
    BuildSchedule
    Compositions
    DecayMode
    Finish
    Info
    InputFiles
    MaterialInfo
    NextIds
    Prototypes
    Recipes
    ResCreators
    Resources
    Snapshots
    StringKeys
    StringVals
    Transactions
    XMLPPInfo

Code Execution: ``-e``
~~~~~~~~~~~~~~~~~~~~~~~
