This is an exemplar map of elections with 8 candidates and 96 voters which requires computing 46056
distances that was generated using the AA heuristic in about 1 minute on Intel Core i9-12900K CPU or
about 7 minutes on Apple M1 (changing AA to BF method in fastmap library results in computing time
of about 2h on Intel Core i9-12900K CPU  ). The reproducible (NOTE: there is some problem with
`seed` parameter for `"norm-mallows"` culture in Mapel library which makes absolute reproducibility
not possible right now) code is in [minimap.py](/tests/minimap.py) file.
 
![alt text](map4071.png "Map of elections using fastmap AA heuristic")
![alt text](map6386.png "Map of elections using fastmap BF")