import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, mapping, LineString
from shapely.ops import unary_union
import numpy as np
from sklearn.cluster import KMeans
import folium

def get_school_coordinates(school_name, city="Reno", state="NV"):
    """Get coordinates for a school using Nominatim geocoding"""
    geolocator = Nominatim(user_agent="school_route_planner")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    search_queries = [
        f"{school_name} High School, {city}, {state}",
        f"{school_name} School, {city}, {state}",
        f"{school_name}, {city}, {state}"
    ]
    
    for query in search_queries:
        try:
            location = geocode(query)
            if location:
                return (location.latitude, location.longitude)
            time.sleep(1)
        except Exception as e:
            print(f"Error geocoding {query}: {e}")
            continue
    
    return None

def calculate_route_length(G, route):
    """Calculate the total length of a route in meters"""
    length = 0
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        if data is not None:
            min_length = min(d.get('length', 0) for d in data.values())
            length += min_length
    return length

def get_walking_distance_areas(G_walk, school_node, zone_geometry, max_walk_time=30):
    """
    Identify areas that are more than max_walk_time minutes from school
    max_walk_time: maximum walking time in minutes
    Returns: area that needs bus stops
    """
    # Average walking speed (meters per minute)
    walking_speed = 80  # ~5 km/h
    max_distance = walking_speed * max_walk_time
    
    # Get all nodes within walking distance
    reachable = nx.single_source_dijkstra_path_length(G_walk, school_node, weight='length')
    walkable_nodes = [node for node, dist in reachable.items() if dist <= max_distance]
    
    # Create a buffer around walkable nodes
    walkable_points = [Point(G_walk.nodes[node]['x'], G_walk.nodes[node]['y']) for node in walkable_nodes]
    walkable_area = unary_union([point.buffer(0.001) for point in walkable_points])  # ~100m buffer
    
    # Get area that needs bus stops (zone minus walkable area)
    if isinstance(zone_geometry, MultiPolygon):
        needs_bus = MultiPolygon([geom for geom in zone_geometry.geoms if not geom.within(walkable_area)])
    else:
        needs_bus = zone_geometry.difference(walkable_area)
    
    return needs_bus

def place_bus_stops(needs_bus_area, G_drive, num_points=20, min_distance_between_stops=200):
    """
    Place bus stops in safe, accessible locations along roads
    
    Parameters:
    - needs_bus_area: Shapely geometry of areas needing bus service
    - G_drive: NetworkX graph of the road network
    - num_points: Target number of bus stops
    - min_distance_between_stops: Minimum distance (meters) between stops
    
    Returns:
    - Array of (lat, lon) coordinates for safe bus stops
    """
    if needs_bus_area.is_empty:
        return np.array([])
    
    # Get all road edges within the area
    road_lines = []
    road_metadata = []
    for u, v, data in G_drive.edges(data=True):
        # Create line from node coordinates
        line = LineString([
            (G_drive.nodes[u]['x'], G_drive.nodes[u]['y']),
            (G_drive.nodes[v]['x'], G_drive.nodes[v]['y'])
        ])
        
        # Only include if the road intersects with the area needing service
        if line.intersects(needs_bus_area):
            # Check road type and speed limit for safety
            highway = data.get('highway', '')
            speed = data.get('maxspeed', 0)
            
            # Skip highways and high-speed roads
            if any(road_type in str(highway).lower() for road_type in 
                  ['motorway', 'trunk', 'primary', 'secondary', 'raceway']):
                continue
            
            # Skip roads with speed limits over 45 mph (roughly 70 km/h)
            try:
                if isinstance(speed, list):
                    speed = float(speed[0])
                elif isinstance(speed, str):
                    speed = float(speed.split()[0])
                if speed > 70:  # km/h
                    continue
            except (ValueError, TypeError):
                # If speed limit can't be parsed, assume it's safe
                pass
            
            road_lines.append(line)
            road_metadata.append(data)
    
    if not road_lines:
        return np.array([])
    
    # Generate points along safe roads
    points = []
    for line in road_lines:
        # Generate points every 50 meters along the road
        distances = np.arange(0, line.length, 50)
        points.extend([list(line.interpolate(distance).coords)[0] for distance in distances])
    
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points)
    if len(points) < num_points:
        return np.array([[y, x] for x, y in points])
    
    # Use KMeans to distribute stops along the safe roads
    kmeans = KMeans(n_clusters=num_points, random_state=42)
    kmeans.fit(points)
    
    # Project cluster centers to nearest safe road
    safe_stops = []
    for center in kmeans.cluster_centers_:
        point = Point(center[0], center[1])
        
        # Find nearest road
        min_dist = float('inf')
        nearest_point = None
        
        for line in road_lines:
            proj_point = nearest_points(point, line)[1]
            dist = point.distance(proj_point)
            
            if dist < min_dist:
                min_dist = dist
                nearest_point = proj_point
        
        if nearest_point:
            safe_stops.append([nearest_point.y, nearest_point.x])
    
    # Remove stops that are too close to each other
    filtered_stops = []
    for stop in safe_stops:
        if not filtered_stops or all(
            Point(stop[1], stop[0]).distance(Point(x[1], x[0])) * 111000 > min_distance_between_stops 
            for x in filtered_stops
        ):
            filtered_stops.append(stop)
    
    return np.array(filtered_stops)

# 1. Load Zoning Data
print("Loading zoning data...")
zoning_data = gpd.read_file("ZoningData/High_School_Zones.geojson")

# 2. Get school coordinates
print("Getting school coordinates...")
school_locations = {}
for school in zoning_data['NAME'].unique():
    coords = get_school_coordinates(school)
    if coords:
        school_locations[school] = coords
        print(f"Found coordinates for {school}: {coords}")
    else:
        print(f"Could not find coordinates for {school}")

# 3. Load both walking and driving networks
print("Loading networks from OSM...")
city = "Reno, United States"
G_walk = ox.graph_from_place(city, network_type="walk")
G_drive = ox.graph_from_place(city, network_type="drive")

# 4. Initialize Folium Map
center_lat = np.mean([coords[0] for coords in school_locations.values()])
center_lon = np.mean([coords[1] for coords in school_locations.values()])
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a color generator for different schools
colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']

# 5. Process each school
for school_idx, school in enumerate(school_locations):
    print(f"Processing routes for {school}...")
    school_color = colors[school_idx % len(colors)]
    
    school_layer = folium.FeatureGroup(name=school)
    
    # Get the zones for the current school
    school_zones = zoning_data[zoning_data['NAME'] == school]
    school_location = school_locations[school]
    
    # Add school marker
    school_layer.add_child(folium.Marker(
        location=school_location,
        popup=f"{school} (School)",
        icon=folium.Icon(color='red', icon='info-sign')
    ))
    
    # Get school node in both networks
    school_node_walk = ox.distance.nearest_nodes(G_walk, school_location[1], school_location[0])
    school_node_drive = ox.distance.nearest_nodes(G_drive, school_location[1], school_location[0])
    
    for idx, zone in school_zones.iterrows():
        zone_geometry = zone.geometry
        
        # Find areas that need bus stops (more than 30 min walk)
        needs_bus_area = get_walking_distance_areas(G_walk, school_node_walk, zone_geometry)
        
        if not needs_bus_area.is_empty:
            # Place bus stops in areas that need them
            stops = place_bus_stops(needs_bus_area)
            
            if len(stops) > 0:
                # Find nearest nodes for stops
                stop_nodes = [ox.distance.nearest_nodes(G_drive, lon, lat) for lat, lon in stops]
                
                # Create routes from each stop to the school
                for i, stop_node in enumerate(stop_nodes):
                    try:
                        route = nx.shortest_path(G_drive, stop_node, school_node_drive, weight='length')
                        route_length = calculate_route_length(G_drive, route)
                        distance_km = route_length / 1000
                        
                        # Add stop marker
                        school_layer.add_child(folium.Marker(
                            location=stops[i].tolist(),
                            popup=f"Bus Stop {i+1} for {school}<br>Distance to school: {distance_km:.2f} km",
                            icon=folium.Icon(color=school_color, icon='bus', prefix='fa')
                        ))
                        
                        # Add route to map
                        route_coords = [(G_drive.nodes[node]['y'], G_drive.nodes[node]['x']) for node in route]
                        school_layer.add_child(folium.PolyLine(
                            route_coords,
                            color=school_color,
                            weight=3,
                            opacity=0.8,
                            popup=f"Route to {school}: {distance_km:.2f} km"
                        ))
                    except nx.NetworkXNoPath:
                        print(f"Could not find route for stop {i+1} to {school}")

    # Add zone boundaries to the map
    folium.GeoJson(
        school_zones,
        style_function=lambda x: {
            'fillColor': school_color,
            'fillOpacity': 0.1,
            'color': school_color,
            'weight': 2
        }
    ).add_to(school_layer)
    
    m.add_child(school_layer)

# 6. Add layer control and save map
folium.LayerControl().add_to(m)
m.save("optimized_bus_routes.html")
print("Map saved as 'optimized_bus_routes.html'")