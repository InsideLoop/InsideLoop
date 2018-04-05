.. role:: cpp(code)

    :language: cpp

Map
===

Overview
--------

The :cpp:`il::Map<K, V>` container implements a hash table that links keys of
type :cpp:`K` to values of type :cpp:`V`. In many languages, this kind of data
structure is called as dictionnary as they connect words to definitions.
The C++ standard library has its own hash table known as
:cpp:`std::unordered_map<K, V>`. We designed our own hash table to get better
performance with an API that we find more natural.

Let's dive with our first example, where we'll map different cities of the world
to their population as of 2018.

.. code-block:: cpp

    #include <il/Map.h>
    #include <il/String.h>

    il::Map<il::String, il::int_t> map{};
    map.Set("Lyon", 506615);
    map.Set("Paris", 2206488);
    map.Set("Boston, 687584);
    map.Set("Beijing", 21707000);

Once this hash table has been filled, it can be used by a function to query for
the population of a given city. In case the city is not available in the map,
it will return :cpp:`-1`. This can be done with the following code.

.. code-block:: cpp

    #include <il/Map.h>
    #include <il/String.h>

    il::int_t population(const il::Map<il::String, il::int_t>& map, const il::String& city) {
      il::spot_t s = map.search(city);
      if (map.found(s)) {
        return map.value(s);
      } else {
        return -1;
      }
    }

To give you another example, if you are a bartender, you might want to map your
clients to the numbers of beers they have bought. The database will be stored
il a :cpp:`il::Map<il::String, il::int_t>` where the key is your client name
and the value is the numbers of beers they have bought. When you
defaut-construct your database :cpp:`map`, it will be empty. Every time one of
your client buy a beer, should call the following function:

.. code-block:: cpp

    #include <il/Map.h>
    #include <il/String.h>

    void addBeer(const il::String& name, il::Map<il::String, il::int_t>& map) {
      il::spot_t s = map.search(name);
      if (map.found(s)) {
        map.Value(s) += 1;
      } else {
        map.Set(name, 1, il::io, s);
      }
    }

If the client has been found in the database, we need to add one beer to the
total of beers he has bought. As you need write access to the value associated
with the key, one should use the method :cpp:`Value` that starts with a capital.
If this is a new client, you only need to set the number of beers to 1. As
the hash of his name has already been computed when we searched for its name in
the :cpp:`map`, it is a good idea to pass the spot parameter :cpp:`s` to the
method so that that hash is not computed again.
