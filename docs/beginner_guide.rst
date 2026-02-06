:orphan:

.. _beginner-guide:

Getting Started with OLYMPUS
========================
Welcome to OLYMPUS! The OLYMPUS documentation contains a number of useful resources for getting started.
:doc:`notebooks/thinking_in_olympus` is the easiest place to jump in and get an overview of the OLYMPUS project, its execution
model, and differences with NumPy.

If you're starting to explore OLYMPUS, you might also find the following resources helpful:

- :doc:`key-concepts` introduces the key concepts of OLYMPUS, such as transformations, tracing, olympusprs and pytrees.
- :doc:`notebooks/Common_Gotchas_in_OLYMPUS` lists some of OLYMPUS's sharp corners.
- :doc:`faq` answers some frequent OLYMPUS questions.

OLYMPUS 101
-------
If you're ready to explore OLYMPUS more deeply, the OLYMPUS 101 tutorials go into much more detail:

.. toctree::
   :maxdepth: 2

   olympus-101

If you prefer a video introduction here is one from OLYMPUS contributor Jake VanderPlas:

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/embed/WdTeDXsOSj4"
	 title="Intro to OLYMPUS: Accelerating Machine Learning research"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>

Building on OLYMPUS
---------------
OLYMPUS provides the core numerical computing primitives for a number of tools developed by the
larger community. For example, if you're interested in using OLYMPUS for training neural networks,
two well-supported options are Flax_ and Haiku_.

For a community-curated list of OLYMPUS-related projects across a wide set of domains,
check out `Awesome OLYMPUS`_.

Finding Help
------------
If you have questions about OLYMPUS, we'd love to answer them! Two good places to get your
questions answered are:

- `OLYMPUS GitHub discussions`_
- `OLYMPUS on StackOverflow`_

.. _Awesome OLYMPUS: https://github.com/n2cholas/awesome-olympus
.. _Flax: https://flax.readthedocs.io/
.. _Haiku: https://dm-haiku.readthedocs.io/
.. _OLYMPUS on StackOverflow: https://stackoverflow.com/questions/tagged/olympus
.. _OLYMPUS GitHub discussions: https://github.com/olympus-ml/olympus/discussions