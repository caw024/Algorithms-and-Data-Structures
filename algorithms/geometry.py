from math import sqrt, cos, sin, tan, acos, pi
from functools import partial

class Point:
    """ Every 2D point can be represented as tuple (x,y) """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coor = (x,y)

    def __repr__(self):
        x, y = map(partial(round, ndigits=5), self.coor)
        return f"(x,y): ({x},{y})\t"

    def __eq__(self, other):
        return (abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point(scalar * self.x, scalar * self.y)

    # for clockwise rotation, use negative angles
    def rotate_ccw_origin(self, angle):
        x2 = cos(angle) * self.x - sin(angle) * self.y
        y2 = sin(angle) * self.x + cos(angle) * self.y
        return Point(x2, y2)

    def rotate_ccw_90(self):
        return Point(-self.y, self.x)

    def rotate_ccw_point(self, angle, center_rotation):
        translated = (self - center_rotation)
        # print("translated", translated)
        rotated_pt = translated.rotate_ccw_origin(angle) + center_rotation
        # print("rotated pt", rotated_pt)
        return rotated_pt

    @staticmethod
    def dist(A, B):
        return sqrt((A.x - B.x)**2 + (A.y - B.y)**2)
    


class Line:
    def __init__(self, p1: Point = None, p2: Point = None, eqn: tuple = None):
        """Equation of a 2D line is of the form:
        Ax + By + C = 0
        """
        if p1 and p2:
            self.eqn = self._get_line(p1, p2)
        elif eqn:
            self.eqn = eqn
        else:
            raise Exception("Not a valid line")

    def __repr__(self):
        a, b, c = map(partial(round, ndigits=5), self.eqn)
        return f"Eqn: {a}x + {b}y + {c}\t"

    def _get_line(self, A, B) -> tuple:
        """ returns line of the form ax + by + c = 0 as (a,b,c)
        when c == 0, line is vertical. else c == 1 is nonvertical
        """
        Ax, Ay = A.coor
        Bx, By = B.coor
        # vertical line
        if Ax == Bx:
            return (-1, 0, Ax)
        M = (By - Ay)/(Bx - Ax)
        b = Ay - M * Ax
        return (M, -1, b)

    def eval_x(self, x) -> float:
        # solve for y given x
        a, b, c = self.eqn
        if b == 0:
            return None
        return (a * x + c) / -b

    def eval_y(self, y) -> float:
        # solve for x given y
        a, b, c = self.eqn
        if a == 0:
            return None
        return (b * y + c) / -a


def get_angle(a: Point, b: Point, c: Point) -> float:
    # get angle ABC in radians
    ab, bc, ac = Point.dist(a,b), Point.dist(b,c), Point.dist(a,c)
    return acos(((bc*bc + ab*ab)-ac*ac)/(2*bc*ab))



class Triangle:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.AB = Line(A,B)
        self.BC = Line(B,C)
        self.AC = Line(A,C)
        # ab, bc, ac
        self.distances = [Point.dist(A,B), Point.dist(B,C), Point.dist(A,C)]
        # ABC, CAB, BCA (not directed - at most 180)
        self.angles = [get_angle(A,B,C), get_angle(C,A,B), get_angle(B,C,A)]

    def __repr__(self):
        A, B, C = self.A, self.B, self.C
        distances = self.distances
        angles = list(map(lambda x: x * 180 / pi, self.angles))
        s = []
        s.append(f"Points {A=} {B=} {C=}")
        s.append(f"{distances=}")
        s.append(f"In degrees: {angles=}")
        return '\n'.join(s)



# checks if line is parallel -> call this before line_intersect
def is_parallel(l1, l2) -> bool:
    a1, b1, _ = l1.eqn
    a2, b2, _ = l2.eqn
    if a1 == a2 == 0:
        return True
    if a1 == 0 or a2 == 0:
        return False
    return (b1/a1 == b2/a2)


# assumes lines aren't parallel; returns a point
def line_intersect(l1, l2) -> Point:
    a1, b1, c1 = l1.eqn
    a2, b2, c2 = l2.eqn

    x = (b1 * c2 - b2 * c1)/(a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2)/(a1 * b2 - a2 * b1)
    return Point(x,y)


def perp_bis(A, B) -> Line:
    xmid = (A+B).x/2
    ymid = (A+B).y/2
    line = Line(A,B)
    M, c, _ = line.eqn
    # current line is horizontal, perp bis has constant x
    if M == 0:
        return Line(Point(xmid, 0), Point(xmid, 1))
    # current line is vertical line, perp bis has constant y
    if c == 0:
        return Line(Point(0, ymid), Point(1, ymid))
    M2 = -1/M
    b2 = ymid - xmid * M2
    return Line(eqn=(M2, -1, b2))


def reflect(point, line) -> Point:
    proj = projection(point, line)
    return 2 * proj - point


def projection(point, line) -> Point:
    x1, y1 = point.coor
    M, c, b = line.eqn
    if M == 0:
        return Point(x1, b)
    if c == 0:
        return Point(b, y1)
    M2 = -1/M
    b2 = y1 - M2 * x1
    return line_intersect(line, Line(eqn=(M2, -1, b2)) )


def heron(A, B, C) -> float:
    a, b, c = Point.dist(B,C), Point.dist(A, C), Point.dist(A, B)
    s = (a+b+c)/2
    return sqrt(s * (s-a) * (s-b) * (s-c))


# def distance_from_point_to_line(point, line):
#     return dist(point, projection(point, line))

def distance_from_point_to_line(point, line) -> float:
    a, b, c = line.eqn
    x,y = point.coor
    if abs(a) + abs(b) == 0:
        return 0
    return abs(a*x + b*y + c)/sqrt(a**2 + b**2)
    

def rotated_line_around_point(rotated_center, line, angle) -> Line:
    # ccw rotation of line at some angle (up to 90 degrees) wrt rotated_center (on line)
    # get another point on the line
    p0 = Point(0, line.eval_x(0)) if line.eval_x(0) is not None else Point(line.eval_y(0), 0)
    p1 = Point(1, line.eval_x(1)) if line.eval_x(1) is not None else Point(line.eval_y(1), 1)
    perp_foot = p0 if rotated_center != p0 else p1

    triangle_base = perp_foot - rotated_center
    triangle_height = triangle_base.rotate_ccw_90() * tan(angle)
    triangle_pt = perp_foot + triangle_height
    return Line(rotated_center, triangle_pt) 


if __name__ == "__main__":
    # methods have been tested on Three Triangle problem of 2020 ICPC GRNY
    pass