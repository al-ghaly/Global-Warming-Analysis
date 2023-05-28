import re
import numpy
import pylab
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]
TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)


class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """

    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}
        # raw temp data
        f = open(filename, 'r')
        # header = ['city', 'temp', 'data'] but capitalized
        header = f.readline().strip().split(',')
        for line in f:
            # items = ['SEATTLE', '3.1', '19610101'] strip to remove new lines and whitespaces from left and right
            items = line.strip().split(',')
            # items[header.index('DATE)] = items[2] formatted (yyyymmdd) G1(4items)G2(2items)G3(2items)
            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))
            # city = items[0]
            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            # rawdata = {'city name' : {year : { month : {day : temperature}}}}
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            # fucked nested nested nested dictionary .
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points x : experimental x
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points y : experimental y
        estimated: an 1-d pylab array of values estimated by a linear
            regression model the y from the model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    # a straightforward mathematical relationship
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y) ** 2).sum()
    var_x = ((x - x.mean()) ** 2).sum()
    SE = pylab.sqrt(EE / (len(x) - 2) / var_x)
    return SE / model[0]


def generate_models(x, y, degrees):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degrees: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models = []
    for degree in degrees:
        models.append(pylab.array(pylab.polyfit(x, y, degree)))
    return models


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.

    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    error = ((y - estimated)**2).sum()
    # just mathematical relationship
    mean = y.sum() / len(y)
    error2 = ((y - mean)**2).sum()
    return 1 - error / error2


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    pylab.plot(x, y, 'bo')
    estimated = pylab.polyval(models[0], x)
    min_r2 = r_squared(y, estimated)
    best_model = models[0]
    for model in models:
        r2 = r_squared(y, pylab.polyval(model, x))
        if r2 <= min_r2:
            min_r2 = r2
            best_model = model
    pylab.plot(x, pylab.polyval(best_model, x), color='red')
    title = 'A fitting curve of degree ' + str(len(best_model) - 1) + '  and r2 of : ' + str(min_r2)
    if len(best_model) == 2:
        se = se_over_slope(x, y, pylab.polyval(best_model, x), best_model)
        title += '\n' + ' and se of ' + str(se)
    pylab.xlabel('years')
    pylab.ylabel('degree in celsius')
    pylab.title(title)
    pylab.show()


# temps = []
# climate = Climate('data.csv')
# for year in TRAINING_INTERVAL:
#     temps.append(climate.get_daily_temp('NEW YORK', 1, 10, year))
# model = generate_models(pylab.array(TRAINING_INTERVAL), pylab.array(temps), [1])[0]
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), pylab.array(temps), [model])
# for year in TRAINING_INTERVAL:
#     temp = climate.get_yearly_temp('NEW YORK', year)
#     temps.append(temp.sum() / len(temp))
# model = generate_models(pylab.array(TRAINING_INTERVAL), pylab.array(temps), [1])[0]
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), pylab.array(temps), [model])
def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    avgs = []
    for year in years:
        temp = []
        for city in multi_cities:
            tmps = climate.get_yearly_temp(city, year)
            temp.append(tmps.sum() / len(tmps))
        temp = pylab.array(temp)
        avgs.append(temp.sum() / len(temp))
    return pylab.array(avgs)


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    temp = y[:]
    for i in range(len(y)):
        avg = 0
        counter = 0
        for j in range(window_length):
            if i - j < 0:
                break
            avg += y[i - j]
            counter += 1
        temp[i] = avg / counter
    return pylab.array(temp)


# model = generate_models(pylab.array(TRAINING_INTERVAL), gen_cities_avg(Climate('data.csv'), CITIES,
#                                                                        list(TRAINING_INTERVAL)), [1])[0]
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL),  gen_cities_avg(Climate('data.csv'), CITIES,
#                                                                             list(TRAINING_INTERVAL)), [model])
# data = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TRAINING_INTERVAL), 5)
# model = generate_models(pylab.array(TRAINING_INTERVAL), data, [1])[0]
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), data, [model])
def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    return ((y - estimated)**2).sum() / len(y)


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    pylab.plot(x, y, 'bo')
    estimated = pylab.polyval(models[0], x)
    min_rmse = rmse(y, estimated)
    best_model = models[0]
    for model in models:
        rmse0 = rmse(y, pylab.polyval(model, x))
        if rmse0 <= min_rmse:
            min_rmse = rmse0
            best_model = model
        pylab.plot(x, pylab.polyval(model, x), color='red')
    title = 'A fitting curve of degree ' + str(len(best_model) - 1) + '  and rmse of : ' + str(min_rmse)
    pylab.xlabel('years')
    pylab.ylabel('degree in celsius')
    pylab.title(title)
    pylab.show()


#
# data = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TRAINING_INTERVAL), 5)
# model = generate_models(pylab.array(TRAINING_INTERVAL), data, [1])[0]
# data = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TESTING_INTERVAL), 3)
# evaluate_models_on_testing(pylab.array(TESTING_INTERVAL), data, [model])
# data = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TRAINING_INTERVAL), 5)
# model = generate_models(pylab.array(TRAINING_INTERVAL), data, [1, 2, 20])
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), data, model)
# model = generate_models(pylab.array(TRAINING_INTERVAL), gen_cities_avg(Climate('data.csv'), CITIES,
#                                                                        list(TRAINING_INTERVAL)), [1, 2, 20])
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL),  gen_cities_avg(Climate('data.csv'), CITIES,
#                                                                             list(TRAINING_INTERVAL)), model)

# data = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TESTING_INTERVAL), 5)
# data1 = moving_average(gen_cities_avg(Climate('data.csv'), CITIES, TRAINING_INTERVAL), 5)
# model = generate_models(pylab.array(TRAINING_INTERVAL), data1, [1, 2, 20])
# evaluate_models_on_testing(pylab.array(TESTING_INTERVAL), data, model)
def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual
        city temperatures for the given cities in a given year.
    """
    sds = []
    for year in years:
        temp = pylab.array([0] * 365)
        for city in multi_cities:
            temp = temp + climate.get_yearly_temp(city, year)[:365]
        temp /= len(multi_cities)
        sds.append(numpy.std(temp))
    return pylab.array(sds)


# data = moving_average(gen_std_devs(Climate('data.csv'), CITIES, TRAINING_INTERVAL), 5)
# model = generate_models(pylab.array(TRAINING_INTERVAL), data, [1])[0]
# evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), data, [model])

